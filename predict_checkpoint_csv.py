#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import tensorflow as tf

from train_efficientnet_b2 import (
    build_model,
    configure_runtime,
    make_dataset,
    normalize_bbox_value,
    sigmoid_np,
)


def load_model(checkpoint: Path) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(checkpoint, compile=False)
    except Exception as error:
        print(f"Full model load failed, trying weight load fallback: {error}")

    model = build_model(learning_rate=1e-5, train_backbone=True, backbone_weights=None)
    model.load_weights(checkpoint)
    return model


def read_prediction_csv(
    input_csv: Path,
    use_bbox_crop: bool,
) -> tuple[list[str], list[str], list[int], list[str]]:
    original_paths = []
    resolved_paths = []
    labels = []
    bboxes = []

    with input_csv.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{input_csv} has no header")
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{input_csv} must contain path,label columns")
        if use_bbox_crop and "bbox" not in reader.fieldnames:
            raise ValueError(f"{input_csv} must contain bbox when --use-bbox-crop is set")

        for row_number, row in enumerate(reader, start=2):
            raw_path = row["path"].strip()
            if not raw_path:
                raise ValueError(f"Empty image path at {input_csv}:{row_number}")

            resolved_path = Path(raw_path)
            if not resolved_path.is_absolute():
                resolved_path = input_csv.parent / resolved_path

            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Label must be 0 or 1 at {input_csv}:{row_number}")

            original_paths.append(raw_path)
            resolved_paths.append(str(resolved_path))
            labels.append(label)
            bboxes.append(
                normalize_bbox_value(
                    row.get("bbox", ""),
                    csv_path=input_csv,
                    row_number=row_number,
                    require_bbox=use_bbox_crop,
                )
            )

    if not original_paths:
        raise ValueError(f"No rows found in {input_csv}")

    return original_paths, resolved_paths, labels, bboxes


def write_scores(output_csv: Path, paths: list[str], labels: list[int], scores) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "label", "score"])
        for path, label, score in zip(paths, labels, scores):
            writer.writerow([path, label, float(score)])


def filter_missing_files(
    original_paths: list[str],
    resolved_paths: list[str],
    labels: list[int],
    bboxes: list[str],
    on_missing: str,
) -> tuple[list[str], list[str], list[int], list[str]]:
    missing = [index for index, path in enumerate(resolved_paths) if not Path(path).exists()]
    if not missing:
        return original_paths, resolved_paths, labels, bboxes

    if on_missing == "raise":
        preview = "\n".join(resolved_paths[index] for index in missing[:20])
        raise FileNotFoundError(f"{len(missing)} input files are missing:\n{preview}")

    missing_set = set(missing)
    print(f"Skipping {len(missing)} missing input files", file=sys.stderr)
    for index in missing[:20]:
        print(f"  missing: {resolved_paths[index]}", file=sys.stderr)

    filtered_original_paths = []
    filtered_resolved_paths = []
    filtered_labels = []
    filtered_bboxes = []
    for index, (original_path, resolved_path, label, bbox) in enumerate(
        zip(original_paths, resolved_paths, labels, bboxes)
    ):
        if index in missing_set:
            continue
        filtered_original_paths.append(original_path)
        filtered_resolved_paths.append(resolved_path)
        filtered_labels.append(label)
        filtered_bboxes.append(bbox)

    if not filtered_original_paths:
        raise FileNotFoundError("All input files are missing; no predictions can be made.")

    return filtered_original_paths, filtered_resolved_paths, filtered_labels, filtered_bboxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run checkpoint inference on a CSV and save path,label,score.")
    parser.add_argument("input_csv", type=Path, help="CSV with path,label columns.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Keras .keras checkpoint.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV with path,label,score.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-bbox-crop", action="store_true", help="Use bbox column crop before resize.")
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--on-missing", choices=["skip", "raise"], default="skip")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime(args.require_gpu, args.mixed_precision)

    original_paths, resolved_paths, labels, bboxes = read_prediction_csv(
        args.input_csv,
        use_bbox_crop=args.use_bbox_crop,
    )
    original_paths, resolved_paths, labels, bboxes = filter_missing_files(
        original_paths,
        resolved_paths,
        labels,
        bboxes,
        args.on_missing,
    )
    dataset = make_dataset(
        paths=resolved_paths,
        labels=labels,
        bboxes=bboxes,
        batch_size=args.batch_size,
        training=False,
        use_bbox_crop=args.use_bbox_crop,
        margin=args.margin,
    )

    model = load_model(args.checkpoint)
    logits = model.predict(dataset).reshape(-1)
    scores = sigmoid_np(logits)
    write_scores(args.output_csv, original_paths, labels, scores)
    print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
