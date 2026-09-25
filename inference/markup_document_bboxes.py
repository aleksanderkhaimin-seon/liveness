#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from run_detector_01 import (
    BOXES_OUTPUT_NAME,
    CLASS_NAMES,
    INPUT_NAME,
    SCORES_OUTPUT_NAME,
    BACKGROUND_MODEL_CLASS_IDS,
    decode_boxes,
    load_onnxruntime,
    nms,
    normalized_to_pixels,
    preprocess_image,
)


def resolve_path(raw_path: str, csv_path: Path) -> Path:
    image_path = Path(raw_path)
    if image_path.is_absolute():
        return image_path
    return csv_path.parent / image_path


class DocumentDetector:
    def __init__(
        self,
        model_path: Path,
        anchors_path: Path,
        threshold: float,
        iou_threshold: float,
        max_detections: int,
        target_label: str,
    ) -> None:
        ort = load_onnxruntime()
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.anchors = np.fromfile(anchors_path, dtype=np.float32).reshape(-1, 4)
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.target_label = target_label

        if target_label not in CLASS_NAMES:
            raise ValueError(f"Unknown label {target_label!r}. Available labels: {CLASS_NAMES}")

    def detect(self, image_path: Path) -> list[dict]:
        _, input_tensor, (image_width, image_height) = preprocess_image(image_path)
        raw_boxes, scores = self.session.run(
            [BOXES_OUTPUT_NAME, SCORES_OUTPUT_NAME],
            {INPUT_NAME: input_tensor},
        )

        raw_boxes = np.squeeze(raw_boxes, axis=0)
        scores = np.squeeze(scores, axis=0)

        if raw_boxes.shape[0] != self.anchors.shape[0]:
            raise RuntimeError(
                f"Model returned {raw_boxes.shape[0]} boxes, but anchors have "
                f"{self.anchors.shape[0]} boxes."
            )

        foreground_class_ids = [
            class_id for class_id in range(len(CLASS_NAMES))
            if class_id not in BACKGROUND_MODEL_CLASS_IDS
        ]

        boxes = decode_boxes(raw_boxes, self.anchors)
        foreground_scores = scores[:, foreground_class_ids]
        best_foreground_offsets = np.argmax(foreground_scores, axis=1)
        best_model_class_ids = np.asarray(foreground_class_ids)[best_foreground_offsets]
        best_scores = scores[np.arange(scores.shape[0]), best_model_class_ids]
        candidate_indices = np.flatnonzero(best_scores >= self.threshold)

        if candidate_indices.size == 0:
            return []

        selected = nms(boxes[candidate_indices], best_scores[candidate_indices], self.iou_threshold)
        detections = []

        for selected_idx in selected:
            anchor_idx = candidate_indices[selected_idx]
            model_class_id = int(best_model_class_ids[anchor_idx])
            label = CLASS_NAMES[model_class_id]
            if label != self.target_label:
                continue

            detections.append(
                {
                    "label": label,
                    "score": float(best_scores[anchor_idx]),
                    "box": normalized_to_pixels(
                        boxes[anchor_idx],
                        image_width=image_width,
                        image_height=image_height,
                    ),
                    "normalized_box": [float(v) for v in boxes[anchor_idx]],
                }
            )

        detections.sort(key=lambda item: item["score"], reverse=True)
        return detections[: self.max_detections]


def read_rows(input_csv: Path) -> tuple[list[dict], list[str]]:
    with input_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{input_csv} has no header")
        if "path" not in reader.fieldnames:
            raise ValueError(f"{input_csv} must contain a path column")
        return list(reader), list(reader.fieldnames)


def write_rows(output_csv: Path, rows: list[dict], fieldnames: list[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_output_path(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_with_bboxes{input_csv.suffix}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Add document detector bbox markup to a CSV.")
    parser.add_argument("input_csv", type=Path, help="CSV with a path column.")
    parser.add_argument("--output-csv", type=Path, help="Output CSV path. Defaults to *_with_bboxes.csv.")
    parser.add_argument("--model", type=Path, default=script_dir / "detector_01.onnx")
    parser.add_argument("--anchors", type=Path, default=script_dir / "anchors.bin")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.6)
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--label", default="document", help="Detector label to write into bbox.")
    parser.add_argument("--bbox-column", default="bbox")
    parser.add_argument("--score-column", help="Optional column for the selected detection score.")
    parser.add_argument(
        "--all-boxes",
        action="store_true",
        help="Write all matching boxes instead of only the highest-score box.",
    )
    parser.add_argument(
        "--on-error",
        choices=["raise", "empty"],
        default="empty",
        help="How to handle unreadable images or inference errors.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv
    output_csv = args.output_csv or default_output_path(input_csv)
    rows, fieldnames = read_rows(input_csv)

    for column in (args.bbox_column, args.score_column):
        if column and column not in fieldnames:
            fieldnames.append(column)

    detector = DocumentDetector(
        model_path=args.model,
        anchors_path=args.anchors,
        threshold=args.threshold,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_detections,
        target_label=args.label,
    )

    for row_index, row in enumerate(rows, start=1):
        image_path = resolve_path(row["path"], input_csv)

        try:
            detections = detector.detect(image_path)
        except Exception as error:
            if args.on_error == "raise":
                raise
            print(f"[{row_index}] {image_path}: {error}", file=sys.stderr)
            detections = []

        if detections:
            if args.all_boxes:
                row[args.bbox_column] = json.dumps(
                    [detection["box"] for detection in detections],
                    separators=(",", ":"),
                )
                if args.score_column:
                    row[args.score_column] = json.dumps(
                        [detection["score"] for detection in detections],
                        separators=(",", ":"),
                    )
            else:
                row[args.bbox_column] = json.dumps(detections[0]["box"], separators=(",", ":"))
                if args.score_column:
                    row[args.score_column] = f"{detections[0]['score']:.8f}"
        else:
            row[args.bbox_column] = ""
            if args.score_column:
                row[args.score_column] = ""

        if args.progress_every > 0 and row_index % args.progress_every == 0:
            print(f"Processed {row_index}/{len(rows)} rows", file=sys.stderr)

    write_rows(output_csv, rows, fieldnames)
    print(f"Saved markup CSV to: {output_csv}")


if __name__ == "__main__":
    main()
