#!/usr/bin/env python3
"""Average the weights of several checkpoints of one run into a new .keras file (SWA-style).

Averaging weights over the late part of training tends to generalise better to unfamiliar
data than any single checkpoint, and it smooths the epoch-to-epoch swings in prod EER, so
the best epoch no longer has to be picked on prod. The checkpoints must share one
architecture -- normally the checkpoints/epoch_*.keras files of a single run.

Floating-point variables are averaged, BatchNormalization moving statistics included.
Integer state (Keras dropout seed generators) is copied from the last checkpoint given.
Averaged BN statistics are an approximation; --bn-csv re-estimates them instead, as SWA
does: one pass of the averaged model over --bn-images frames of that CSV, with validation
preprocessing (no augmentation) and a cumulative mean rather than the layers' momentum.

The output keeps the inputs' layer names, so predict_checkpoint_csv.py,
convert_checkpoint_to_onnx.py and launch_eval.py --checkpoint take it like any checkpoint.
Put it inside the run folder (e.g. <run>/averaged/) so the run's report.json is found for
the input geometry. <output>.json next to it records the inputs, weights and BN pass.

Examples:
  python average_checkpoints.py runs/x/checkpoints/epoch_002_*.keras runs/x/checkpoints/epoch_003_*.keras \\
      --output runs/x/averaged/avg_e2_e3.keras
  python average_checkpoints.py runs/x/checkpoints --output runs/x/averaged/avg_all_bn.keras \\
      --bn-csv data/train.csv --bn-images 2000
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from input_geometry import find_report, resolve_geometry
from predict_checkpoint_csv import filter_missing_files, load_model, read_prediction_csv
from progress_util import progress
from train_efficientnet_b2 import configure_runtime, make_dataset, set_input_geometry


def expand_inputs(items: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        if item.is_dir():
            found = sorted(item.glob("*.keras"))
            if not found:
                raise SystemExit(f"No .keras files in {item}")
            paths.extend(found)
        elif item.is_file():
            paths.append(item)
        else:
            raise SystemExit(f"Not found: {item}")
    if len(paths) < 2:
        raise SystemExit(f"Need at least two checkpoints to average, got {len(paths)}")
    return paths


def iter_layers(model):
    for layer in model.layers:
        yield layer
        if hasattr(layer, "layers"):
            yield from iter_layers(layer)


def is_float(variable) -> bool:
    return tf.as_dtype(variable.dtype).is_floating


def average_weights(paths: list[Path], weights: np.ndarray) -> tuple[tf.keras.Model, dict]:
    """Return the last checkpoint's model with its float variables replaced by the weighted mean."""
    sums: list[np.ndarray | None] = []
    reference: list[tuple[str, tuple]] = []
    model = None
    for k, path in enumerate(paths):
        if model is not None:
            del model
            gc.collect()
        print(f"Loading {path} (weight {weights[k]:.3f})")
        model = load_model(path)
        variables = model.weights
        signature = [(v.path, tuple(v.shape)) for v in variables]
        if k == 0:
            reference = signature
            sums = [np.zeros(v.shape, np.float64) if is_float(v) else None for v in variables]
        elif signature != reference:
            mismatch = next(i for i, (a, b) in enumerate(zip(signature, reference)) if a != b) \
                if len(signature) == len(reference) else None
            detail = f"first difference: {signature[mismatch]} vs {reference[mismatch]}" if mismatch is not None \
                else f"{len(signature)} vs {len(reference)} variables"
            raise SystemExit(f"{path} does not match {paths[0]}: {detail}")
        for total, variable in zip(sums, variables):
            if total is not None:
                total += weights[k] * np.asarray(variable.numpy(), np.float64)

    averaged = copied = 0
    for total, variable in zip(sums, model.weights):
        if total is None:
            copied += 1  # integer state stays as in the last checkpoint
        else:
            variable.assign(tf.cast(total / weights.sum(), variable.dtype))
            averaged += 1
    return model, {"averaged_variables": averaged, "copied_variables": copied}


def recompute_bn(model: tf.keras.Model, args) -> dict:
    """Re-estimate BN moving statistics with a cumulative mean over frames of args.bn_csv."""
    layers = [layer for layer in iter_layers(model)
              if isinstance(layer, tf.keras.layers.BatchNormalization) and layer.trainable]
    if not layers:
        print("No trainable BatchNormalization layers (backbone was frozen); BN statistics left as averaged")
        return {"layers": 0}
    _, resolved, labels, bboxes = read_prediction_csv(args.bn_csv, use_bbox_crop=args.use_bbox_crop)
    _, resolved, labels, bboxes = filter_missing_files(resolved, resolved, labels, bboxes, "skip")
    rng = np.random.default_rng(args.seed)
    keep = np.sort(rng.choice(len(resolved), size=min(args.bn_images, len(resolved)), replace=False))
    paths = [resolved[i] for i in keep]
    dataset = make_dataset(paths, [labels[i] for i in keep], [bboxes[i] for i in keep], args.batch_size,
                           training=False, use_bbox_crop=args.use_bbox_crop, margin=args.margin)

    momenta = [layer.momentum for layer in layers]
    for layer in layers:
        layer.moving_mean.assign(tf.zeros_like(layer.moving_mean))
        layer.moving_variance.assign(tf.ones_like(layer.moving_variance))
    batches = 0
    try:
        total = int(np.ceil(len(paths) / args.batch_size))
        for images, _ in progress(dataset, total, f"Re-estimating BN statistics on {len(paths)} frames"):
            # Keras: moving = moving * momentum + batch * (1 - momentum); k / (k + 1) gives the running mean.
            for layer in layers:
                layer.momentum = batches / (batches + 1)
            model(images, training=True)  # eager, so each call reads the momentum set above
            batches += 1
    finally:
        for layer, momentum in zip(layers, momenta):
            layer.momentum = momentum
    return {"layers": len(layers), "csv": str(args.bn_csv), "frames": len(paths), "batches": batches}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", type=Path, nargs="+", help=".keras files, or directories of them (non-recursive).")
    ap.add_argument("--output", type=Path, required=True, help="Averaged .keras file to write.")
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="One weight per checkpoint, in the order given (default: equal).")
    ap.add_argument("--bn-csv", type=Path, default=None,
                    help="Training manifest to re-estimate BatchNormalization statistics on (path,label[,bbox]).")
    ap.add_argument("--bn-images", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--use-bbox-crop", action="store_true", help="Same as the run, for the BN pass.")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--resize-mode", choices=("squash", "letterbox"), default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--require-gpu", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime(args.require_gpu, mixed_precision=False)
    paths = expand_inputs(args.checkpoints)
    weights = np.ones(len(paths)) if args.weights is None else np.asarray(args.weights, np.float64)
    if len(weights) != len(paths):
        raise SystemExit(f"--weights has {len(weights)} values for {len(paths)} checkpoints")
    if (weights < 0).any() or weights.sum() <= 0:
        raise SystemExit("--weights must be non-negative with a positive sum")

    # load_model's weight-load fallback builds at the module geometry, so set it first (as predict_checkpoint_csv does).
    report = find_report(paths[0])
    geometry = resolve_geometry(args.image_size, args.resize_mode, report_path=report)
    set_input_geometry(geometry.image_size, geometry.resize_mode)

    model, stats = average_weights(paths, weights)
    geometry = resolve_geometry(args.image_size, args.resize_mode, model_input_size=int(model.input_shape[1]),
                                report_path=report)
    set_input_geometry(geometry.image_size, geometry.resize_mode)
    print(f"Averaged {stats['averaged_variables']} variables, copied {stats['copied_variables']} integer ones; "
          f"input geometry {geometry}")
    bn = recompute_bn(model, args) if args.bn_csv else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    record = {
        "output": str(args.output),
        "checkpoints": [str(path) for path in paths],
        "weights": (weights / weights.sum()).round(6).tolist(),
        "bn_recompute": bn,
        "input": {"image_size": geometry.image_size, "resize_mode": geometry.resize_mode},
        **stats,
    }
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
