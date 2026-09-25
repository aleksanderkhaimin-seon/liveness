#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import tensorflow as tf

from train_efficientnet_b2 import (
    build_model,
    compute_eer_metrics,
    configure_runtime,
    make_dataset,
    normalize_bbox_value,
    sigmoid_np,
)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Expected s3://bucket/key path, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def local_path_for_s3(uri: str, cache_dir: Path) -> Path:
    bucket, key = parse_s3_uri(uri)
    return cache_dir / bucket / key


def run_command(command: list[str], dry_run: bool = False) -> None:
    print("$ " + " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def sync_prefix(prefix: str, cache_dir: Path, dry_run: bool = False) -> None:
    destination = local_path_for_s3(prefix.rstrip("/") + "/__prefix_placeholder__", cache_dir).parent
    destination.mkdir(parents=True, exist_ok=True)
    run_command(["aws", "s3", "sync", prefix.rstrip("/") + "/", str(destination)], dry_run=dry_run)


def copy_s3_file(uri: str, local_path: Path, dry_run: bool = False) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(["aws", "s3", "cp", uri, str(local_path)], dry_run=dry_run)


def read_eval_csv(input_csv: Path, cache_dir: Path, use_bbox_crop: bool) -> tuple[list[dict], list[str], list[int], list[str]]:
    rows = []
    paths = []
    labels = []
    bboxes = []

    with input_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{input_csv} has no header")
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{input_csv} must contain path,label columns")
        if use_bbox_crop and "bbox" not in reader.fieldnames:
            raise ValueError(f"{input_csv} must contain bbox when --use-bbox-crop is set")

        for row_number, row in enumerate(reader, start=2):
            raw_path = row["path"].strip()
            if raw_path.startswith("s3://"):
                local_path = local_path_for_s3(raw_path, cache_dir)
            else:
                local_path = Path(raw_path)
                if not local_path.is_absolute():
                    local_path = input_csv.parent / local_path

            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Label must be 0 or 1 at {input_csv}:{row_number}")

            rows.append(row)
            paths.append(str(local_path))
            labels.append(label)
            bboxes.append(
                normalize_bbox_value(
                    row.get("bbox", ""),
                    csv_path=input_csv,
                    row_number=row_number,
                    require_bbox=use_bbox_crop,
                )
            )

    if not rows:
        raise ValueError(f"No rows found in {input_csv}")

    return rows, paths, labels, bboxes


def unique_s3_paths(rows: list[dict]) -> list[str]:
    return sorted({row["path"].strip() for row in rows if row["path"].strip().startswith("s3://")})


def sync_inputs(
    rows: list[dict],
    cache_dir: Path,
    sync_prefixes: list[str],
    copy_missing: bool,
    dry_run: bool,
) -> None:
    for prefix in sync_prefixes:
        sync_prefix(prefix, cache_dir, dry_run=dry_run)

    if not copy_missing:
        return

    for s3_path in unique_s3_paths(rows):
        local_path = local_path_for_s3(s3_path, cache_dir)
        if not local_path.exists():
            copy_s3_file(s3_path, local_path, dry_run=dry_run)


def load_model(checkpoint: Path) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(checkpoint, compile=False)
    except Exception as error:
        print(f"Full model load failed, trying weight load fallback: {error}")

    model = build_model(learning_rate=1e-5, train_backbone=True, backbone_weights=None)
    model.load_weights(checkpoint)
    return model


def compile_for_eval(model: tf.keras.Model) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0),
            tf.keras.metrics.AUC(name="auc", from_logits=True),
            tf.keras.metrics.Precision(name="precision", thresholds=0.0),
            tf.keras.metrics.Recall(name="recall", thresholds=0.0),
        ],
    )


def predict(
    model: tf.keras.Model,
    paths: list[str],
    labels: list[int],
    bboxes: list[str],
    batch_size: int,
    use_bbox_crop: bool,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = make_dataset(
        paths=paths,
        labels=labels,
        bboxes=bboxes,
        batch_size=batch_size,
        training=False,
        use_bbox_crop=use_bbox_crop,
        margin=margin,
    )
    logits = model.predict(dataset).reshape(-1)
    return logits, sigmoid_np(logits)


def write_predictions(
    output_csv: Path,
    source_rows: list[dict],
    local_paths: list[str],
    logits: np.ndarray,
    scores: np.ndarray,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(source_rows[0].keys())
    for column in ("local_path", "logit", "score", "prediction"):
        if column not in fieldnames:
            fieldnames.append(column)

    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row, local_path, logit, score in zip(source_rows, local_paths, logits, scores):
            output_row = dict(row)
            output_row["local_path"] = local_path
            output_row["logit"] = float(logit)
            output_row["score"] = float(score)
            output_row["prediction"] = int(logit >= 0.0)
            writer.writerow(output_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync S3 dataframe images locally and evaluate a liveness checkpoint.")
    parser.add_argument("input_csv", type=Path, help="CSV with path,label columns. path may be s3://bucket/key.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Keras .keras checkpoint to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/s3_eval"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/mnt/userefs/aleksandr_khaimin/Work/liveness/s3_cache"))
    parser.add_argument("--sync-prefix", action="append", default=[], help="Optional s3://bucket/prefix to sync before per-file copy. Can be repeated.")
    parser.add_argument("--no-copy-missing", action="store_true", help="Only run --sync-prefix commands; do not aws s3 cp missing files.")
    parser.add_argument("--dry-run", action="store_true", help="Print AWS commands without running them.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eer-threshold", type=float, default=0.5)
    parser.add_argument("--use-bbox-crop", action="store_true")
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime(args.require_gpu, args.mixed_precision)

    rows, paths, labels, bboxes = read_eval_csv(args.input_csv, args.cache_dir, args.use_bbox_crop)
    sync_inputs(
        rows=rows,
        cache_dir=args.cache_dir,
        sync_prefixes=args.sync_prefix,
        copy_missing=not args.no_copy_missing,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("Dry run complete. No evaluation was run.")
        return

    missing_paths = [path for path in paths if not Path(path).exists()]
    if missing_paths:
        preview = "\n".join(missing_paths[:20])
        raise FileNotFoundError(f"{len(missing_paths)} local files are missing after sync/copy:\n{preview}")

    model = load_model(args.checkpoint)
    compile_for_eval(model)
    logits, scores = predict(
        model=model,
        paths=paths,
        labels=labels,
        bboxes=bboxes,
        batch_size=args.batch_size,
        use_bbox_crop=args.use_bbox_crop,
        margin=args.margin,
    )
    metrics = model.evaluate(
        make_dataset(paths, labels, bboxes, args.batch_size, False, args.use_bbox_crop, args.margin),
        return_dict=True,
    )
    eer_metrics = compute_eer_metrics(labels, scores, args.eer_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = args.output_dir / "predictions.csv"
    write_predictions(predictions_csv, rows, paths, logits, scores)

    report = {
        "input_csv": str(args.input_csv),
        "checkpoint": str(args.checkpoint),
        "cache_dir": str(args.cache_dir),
        "samples": len(labels),
        "metrics": metrics,
        "eer_metrics": eer_metrics,
        "bbox_crop": {
            "enabled": args.use_bbox_crop,
            "margin": args.margin,
        },
        "predictions_csv": str(predictions_csv),
    }
    report_path = args.output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
