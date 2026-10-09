#!/usr/bin/env python3
"""Look for frequency shortcuts: label information in the image spectrum that exists in
the training data but not in production, and whether a trained model leans on it.

Everything is measured on the network input exactly as validation sees it (geometry,
bbox crop and --degrade as in make_dataset(training=False)).

Part 1, no model -- the spectrum alone:
  Each image's luma power spectrum (Hann window, DC dropped) is pooled into --bands
  radial bands (0 = DC .. 1 = Nyquist) x 4 orientations, logged, and its per-image
  mean removed, so only spectral *shape* is left, not overall contrast.

  probe_auc     linear probe attack-vs-live on those features:
                  train_cv  group-cross-validated within --train (sessions kept apart)
                  prod      fitted on all of --train, scored on --prod
                  test      ditto on --test, if given
                  prod_cv   fitted and cross-validated on --prod itself: how much label
                            signal the spectrum genuinely carries in production
                A frequency shortcut looks like train_cv >> prod_cv, with prod near 0.5
                (or below 0.5: the cue is reversed in production).
  domain_auc    per label, spectrum-only probe train-vs-prod (0.5 = same spectra).
  effect sizes  per (band, orientation) cell: Cohen's d of attack minus live in each
                split. `lost` = how much of the train effect is missing in production
                (|d_train| - d_prod * sign(d_train)); cells with |d_train| >= 0.5 that
                keep under a quarter of it, or flip sign, are flagged. effect_sizes.csv,
                ranked by `lost`; spectra.png draws them.

Part 2, with --checkpoint -- what the model uses:
  Every split is scored again with one radial band removed at a time (DC kept; the
  last band runs into the corners). band_reliance.csv has EER per band and split and
  the change against the unfiltered input. A band whose removal costs EER on --test
  (in-distribution) but not on --prod, or helps prod, is one the model relies on and
  that does not transfer -- the band to suppress or randomise in training (--freq-aug).

Orientation names refer to the frequency direction: `horiz` = frequency along x
(vertical edges, e.g. text strokes), `vert` = along y, `diag45` / `diag135`.

Example (Code Editor, sees EFS):
  python frequency_shortcuts.py \\
      --train data/train_Seon-PintV1_..._balanced.csv --prod data/ProdTest-0.3.csv \\
      --test data/test_Pinterest_v_1_checked.csv \\
      --checkpoint eval_results/<job>/output/best.keras --out-dir runs/freq/<job>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from eer import compute_eer
from embedding_gap import sample_rows
from input_geometry import find_report, resolve_geometry
from mine_hard_samples import derive_group
from predict_checkpoint_csv import filter_missing_files, load_model, read_prediction_csv
from progress_util import progress
import train_efficientnet_b2 as tr

ORIENTATIONS = ("horiz", "diag45", "vert", "diag135")


# -- spectral features --------------------------------------------------------

def cell_index(size: int, bands: int) -> np.ndarray:
    """(size, size) map from fft2 bin to band * 4 + orientation; -1 for DC and beyond Nyquist."""
    fy = np.fft.fftfreq(size)[:, np.newaxis]
    fx = np.fft.fftfreq(size)[np.newaxis, :]
    rho = np.sqrt(fy**2 + fx**2) / 0.5
    theta = np.mod(np.arctan2(fy, fx), np.pi)  # a real image's spectrum is point-symmetric
    orient = np.round(theta / (np.pi / 4)).astype(int) % 4
    band = np.minimum((rho * bands).astype(int), bands - 1)
    index = band * 4 + orient
    index[(rho == 0) | (rho > 1.0)] = -1
    return index


def spectral_features(images: np.ndarray, index: np.ndarray, n_cells: int) -> np.ndarray:
    luma = images @ np.array([0.299, 0.587, 0.114], np.float32)
    size = luma.shape[1]
    window = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    flat = index.reshape(-1)
    keep = flat >= 0
    counts = np.bincount(flat[keep], minlength=n_cells)
    feats = np.empty((len(luma), n_cells), np.float64)
    for i, image in enumerate(luma):
        image = (image - image.mean()) * window
        power = np.abs(np.fft.fft2(image)) ** 2
        sums = np.bincount(flat[keep], weights=power.reshape(-1)[keep], minlength=n_cells)
        feats[i] = np.log(sums / np.maximum(counts, 1) + 1e-6)
    return feats - feats.mean(axis=1, keepdims=True)


def band_gains(size: int, bands: int) -> np.ndarray:
    """(bands, size, size // 2 + 1) masks on the rfft2d grid, each removing one radial band, DC kept."""
    rho = tr.radial_frequency(size)
    gains = np.ones((bands, *rho.shape), np.float32)
    for k in range(bands):
        upper = (k + 1) / bands if k < bands - 1 else np.inf
        gains[k][(rho >= k / bands) & (rho < upper)] = 0.0
    gains[:, 0, 0] = 1.0
    return gains


# -- one pass over a split ------------------------------------------------------

def scan(csv_path: Path, args, index: np.ndarray, n_cells: int, model, gains) -> dict:
    orig, resolved, labels, bboxes = read_prediction_csv(csv_path, use_bbox_crop=args.use_bbox_crop)
    orig, resolved, labels, bboxes = filter_missing_files(orig, resolved, labels, bboxes, "skip")
    keep = sample_rows(resolved, labels, args.max_per_class, args.seed)
    paths = [resolved[i] for i in keep]
    bboxes = [bboxes[i] for i in keep]
    labels = np.asarray([labels[i] for i in keep])
    dataset = tr.make_dataset(paths, labels.tolist(), bboxes, args.batch_size, training=False,
                              use_bbox_crop=args.use_bbox_crop, margin=args.margin)
    feats, logits = [], []
    gain_tensor = None if gains is None else tf.constant(gains)
    total = int(np.ceil(len(paths) / args.batch_size))
    for images, _ in progress(dataset, total, f"Scanning {csv_path.name} ({len(paths)} images)"):
        feats.append(spectral_features(images.numpy(), index, n_cells))
        if model is not None:
            row = [model(images, training=False).numpy().reshape(-1)]
            for k in range(len(gains)):
                filtered = tr.apply_spectral_gain(images, gain_tensor[k])
                row.append(model(filtered, training=False).numpy().reshape(-1))
            logits.append(np.stack(row, axis=1))
    return {
        "paths": np.asarray(paths),
        "labels": labels,
        "groups": np.asarray([derive_group(p) for p in paths]),
        "feats": np.concatenate(feats),
        "logits": np.concatenate(logits) if logits else None,
    }


# -- statistics -----------------------------------------------------------------

def probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))


def cv_auc(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

    if len(np.unique(y)) < 2:
        return None
    folds = GroupKFold(5) if len(np.unique(groups)) >= 5 else StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(probe(), x, y, cv=folds, groups=groups, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def transfer_auc(train: dict, other: dict) -> float:
    from sklearn.metrics import roc_auc_score

    model = probe().fit(train["feats"], train["labels"])
    return float(roc_auc_score(other["labels"], model.predict_proba(other["feats"])[:, 1]))


def cohens_d(feats: np.ndarray, labels: np.ndarray) -> np.ndarray:
    a, l = feats[labels == 1], feats[labels == 0]
    pooled = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(l) - 1) * l.var(0, ddof=1)) / (len(a) + len(l) - 2))
    return (a.mean(0) - l.mean(0)) / np.maximum(pooled, 1e-9)


def eer_and_auc(labels: np.ndarray, logits: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    scores = tr.sigmoid_np(logits)
    eer, _ = compute_eer(tar=scores[labels == 1], imp=scores[labels == 0])
    return float(eer), float(roc_auc_score(labels, scores))


def cell_names(bands: int) -> list[tuple[float, float, str]]:
    return [(k / bands, (k + 1) / bands, o) for k in range(bands) for o in ORIENTATIONS]


def plot(splits: dict[str, dict], d: dict[str, np.ndarray], bands: int, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping spectra.png")
        return
    centres = (np.arange(bands) + 0.5) / bands
    colors = {"train": "tab:blue", "prod": "tab:red", "test": "tab:green"}
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for j, orient in enumerate(ORIENTATIONS):
        for name, split in splits.items():
            for label, style in ((0, "-"), (1, "--")):
                rows = split["feats"][split["labels"] == label][:, j::4]
                axes[0, j].plot(centres, rows.mean(0), style, color=colors.get(name, "k"),
                                label=f"{name} {'attack' if label else 'live'}")
            axes[1, j].plot(centres, d[name][j::4], color=colors.get(name, "k"), label=name)
        axes[0, j].set_title(f"{orient}: mean log power (shape)")
        axes[1, j].set_title(f"{orient}: Cohen's d, attack - live")
        axes[1, j].axhline(0, color="grey", lw=0.8)
        axes[1, j].set_xlabel("frequency / Nyquist")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


# -- main -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True, help="Training manifest (path,label[,bbox]).")
    ap.add_argument("--prod", type=Path, required=True, help="Production manifest, e.g. data/ProdTest-0.3.csv.")
    ap.add_argument("--test", type=Path, default=None, help="In-distribution test manifest, e.g. Pinterest test.")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Also measure which bands this model relies on.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bands", type=int, default=16, help="Radial bands for the spectral features.")
    ap.add_argument("--reliance-bands", type=int, default=8, help="Radial bands removed one at a time for --checkpoint.")
    ap.add_argument("--max-per-class", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-bbox-crop", action="store_true")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--degrade", default="", help="Same --degrade the model was trained with, if any.")
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--resize-mode", choices=tr.RESIZE_MODES, default=None)
    ap.add_argument("--require-gpu", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    tr.configure_runtime(args.require_gpu, mixed_precision=False)
    tr.DEGRADATIONS[:] = tr.parse_degrade(args.degrade)
    model = None
    if args.checkpoint is not None:
        report = find_report(args.checkpoint)
        model = load_model(args.checkpoint)
        geo = resolve_geometry(args.image_size, args.resize_mode, model_input_size=int(model.input_shape[1]),
                               report_path=report)
    else:
        geo = resolve_geometry(args.image_size, args.resize_mode)
    tr.set_input_geometry(geo.image_size, geo.resize_mode)
    print(f"Input geometry: {geo}")

    index = cell_index(tr.IMAGE_SIZE, args.bands)
    n_cells = args.bands * 4
    gains = band_gains(tr.IMAGE_SIZE, args.reliance_bands) if model is not None else None
    splits = {"train": args.train, "prod": args.prod}
    if args.test is not None:
        splits["test"] = args.test
    data = {name: scan(path, args, index, n_cells, model, gains) for name, path in splits.items()}

    train, prod = data["train"], data["prod"]
    probe_auc = {
        "train_cv": cv_auc(train["feats"], train["labels"], train["groups"]),
        "prod": transfer_auc(train, prod),
        "prod_cv": cv_auc(prod["feats"], prod["labels"], prod["groups"]),
    }
    if "test" in data:
        probe_auc["test"] = transfer_auc(train, data["test"])
    domain_auc = {}
    for label, name in ((0, "live"), (1, "attack")):
        a = train["feats"][train["labels"] == label]
        b = prod["feats"][prod["labels"] == label]
        if len(a) > 20 and len(b) > 20:
            x = np.vstack([a, b])
            y = np.r_[np.zeros(len(a)), np.ones(len(b))]
            groups = np.r_[train["groups"][train["labels"] == label], prod["groups"][prod["labels"] == label]]
            domain_auc[name] = cv_auc(x, y, groups)

    d = {name: cohens_d(split["feats"], split["labels"]) for name, split in data.items()}
    lost = np.abs(d["train"]) - d["prod"] * np.sign(d["train"])
    flagged = (np.abs(d["train"]) >= 0.5) & (d["prod"] * np.sign(d["train"]) < 0.25 * np.abs(d["train"]))
    cells = cell_names(args.bands)
    order = np.argsort(-lost)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "effect_sizes.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["band_lo", "band_hi", "orientation", *(f"d_{n}" for n in d), "lost", "shortcut"])
        for i in order:
            lo, hi, orient = cells[i]
            writer.writerow([f"{lo:.3f}", f"{hi:.3f}", orient, *(f"{d[n][i]:.3f}" for n in d),
                             f"{lost[i]:.3f}", int(flagged[i])])

    summary = {
        "train": str(args.train), "prod": str(args.prod), "test": str(args.test) if args.test else None,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "input": {"image_size": tr.IMAGE_SIZE, "resize_mode": tr.RESIZE_MODE, "degrade": args.degrade},
        "n": {name: {"live": int((s["labels"] == 0).sum()), "attack": int((s["labels"] == 1).sum())}
              for name, s in data.items()},
        "probe_auc": probe_auc,
        "domain_auc": domain_auc,
        "flagged_cells": int(flagged.sum()),
        "top_lost_cells": [
            {"band": [cells[i][0], cells[i][1]], "orientation": cells[i][2],
             **{f"d_{n}": round(float(d[n][i]), 3) for n in d}, "lost": round(float(lost[i]), 3)}
            for i in order[:10]
        ],
    }

    if model is not None:
        rows = []
        reliance = {}
        for name, split in data.items():
            base_eer, base_auc = eer_and_auc(split["labels"], split["logits"][:, 0])
            reliance[name] = {"none": round(base_eer, 3)}
            rows.append([name, "none", "", f"{base_eer:.3f}", f"{base_auc:.4f}", "0.000"])
            for k in range(args.reliance_bands):
                eer, auc = eer_and_auc(split["labels"], split["logits"][:, k + 1])
                band = f"{k / args.reliance_bands:.3f}-{(k + 1) / args.reliance_bands:.3f}"
                reliance[name][band] = round(eer - base_eer, 3)
                rows.append([name, band, k, f"{eer:.3f}", f"{auc:.4f}", f"{eer - base_eer:.3f}"])
        with (args.out_dir / "band_reliance.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["split", "removed_band", "band_index", "eer", "auc", "d_eer"])
            writer.writerows(rows)
        summary["band_reliance_d_eer"] = reliance

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(args.out_dir / "features.npz",
                        **{f"{n}_{k}": v for n, s in data.items() for k, v in s.items() if v is not None})
    plot(data, d, args.bands, args.out_dir / "spectra.png")

    print("\nprobe AUC (spectrum only, attack vs live):",
          {k: (round(v, 3) if v is not None else None) for k, v in probe_auc.items()})
    print("domain AUC train vs prod (spectrum only):",
          {k: (round(v, 3) if v is not None else None) for k, v in domain_auc.items()})
    print(f"{int(flagged.sum())} of {n_cells} cells flagged as shortcut candidates; top by lost effect:")
    for row in summary["top_lost_cells"][:5]:
        print("  ", row)
    if model is not None:
        print("dEER (percentage points) when a band is removed:")
        for name, values in summary["band_reliance_d_eer"].items():
            print(f"  {name}: {values}")


if __name__ == "__main__":
    main()
