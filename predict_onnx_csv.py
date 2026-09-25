#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_SIZE = 512


# ── CSV helpers (mirrors predict_checkpoint_csv.py) ──────────────────────────

def normalize_bbox_value(raw_bbox: str, csv_path: Path, row_number: int, require_bbox: bool) -> str:
    bbox = raw_bbox.strip()
    if bbox.lower() in {"", "none", "null", "nan"}:
        return ""
    try:
        values = json.loads(bbox)
    except json.JSONDecodeError as error:
        if not require_bbox:
            return bbox
        raise ValueError(f"Invalid bbox JSON at {csv_path}:{row_number}: {bbox!r}") from error
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"bbox must be [x1,y1,x2,y2] at {csv_path}:{row_number}: {bbox!r}")
    try:
        x1, y1, x2, y2 = [float(v) for v in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"bbox values must be numeric at {csv_path}:{row_number}: {bbox!r}") from error
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError(f"bbox values must be finite at {csv_path}:{row_number}: {bbox!r}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox must satisfy x2>x1 and y2>y1 at {csv_path}:{row_number}: {bbox!r}")
    return json.dumps([x1, y1, x2, y2], separators=(",", ":"))


def read_prediction_csv(
    input_csv: Path,
    use_bbox_crop: bool,
) -> tuple[list[str], list[str], list[int], list[str]]:
    original_paths, resolved_paths, labels, bboxes = [], [], [], []

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


def filter_missing_files(
    original_paths: list[str],
    resolved_paths: list[str],
    labels: list[int],
    bboxes: list[str],
    on_missing: str,
) -> tuple[list[str], list[str], list[int], list[str]]:
    missing = [i for i, p in enumerate(resolved_paths) if not Path(p).exists()]
    if not missing:
        return original_paths, resolved_paths, labels, bboxes

    if on_missing == "raise":
        preview = "\n".join(resolved_paths[i] for i in missing[:20])
        raise FileNotFoundError(f"{len(missing)} input files are missing:\n{preview}")

    missing_set = set(missing)
    print(f"Skipping {len(missing)} missing input files", file=sys.stderr)
    for i in missing[:20]:
        print(f"  missing: {resolved_paths[i]}", file=sys.stderr)

    out = [(op, rp, lb, bx) for i, (op, rp, lb, bx)
           in enumerate(zip(original_paths, resolved_paths, labels, bboxes))
           if i not in missing_set]
    if not out:
        raise FileNotFoundError("All input files are missing; no predictions can be made.")
    return map(list, zip(*out))


def write_scores(output_csv: Path, paths: list[str], labels: list[int], scores: np.ndarray) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "label", "score"])
        for path, label, score in zip(paths, labels, scores):
            writer.writerow([path, label, float(score)])


# ── Image loading ─────────────────────────────────────────────────────────────

def crop_by_bbox(img: Image.Image, bbox_str: str, margin: float) -> Image.Image:
    if not bbox_str:
        return img
    x1, y1, x2, y2 = json.loads(bbox_str)
    w, h = img.size
    bw, bh = x2 - x1, y2 - y1
    m = margin / 100.0
    x1 = max(0.0, x1 - bw * m)
    y1 = max(0.0, y1 - bh * m)
    x2 = min(float(w), x2 + bw * m)
    y2 = min(float(h), y2 + bh * m)
    return img.crop((int(math.floor(x1)), int(math.floor(y1)),
                     int(math.ceil(x2)),  int(math.ceil(y2))))


def load_image(path: str, bbox: str, use_bbox_crop: bool, margin: float) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if use_bbox_crop:
        img = crop_by_bbox(img, bbox, margin)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.float32)  # [H, W, 3] in [0, 255]


# ── Inference ─────────────────────────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict(
    session: ort.InferenceSession,
    paths: list[str],
    bboxes: list[str],
    batch_size: int,
    use_bbox_crop: bool,
    margin: float,
) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    all_scores = []

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        batch_bboxes = bboxes[start:start + batch_size]

        images = []
        for path, bbox in zip(batch_paths, batch_bboxes):
            try:
                images.append(load_image(path, bbox, use_bbox_crop, margin))
            except Exception as e:
                print(f"Warning: failed to load {path}: {e}", file=sys.stderr)
                images.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32))

        batch = np.stack(images, axis=0)  # [B, H, W, 3]
        logits = session.run(None, {input_name: batch})[0].reshape(-1)
        all_scores.append(logits)

        done = min(start + batch_size, len(paths))
        print(f"\r{done}/{len(paths)}", end="", flush=True)

    print()
    return np.concatenate(all_scores)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX inference on a CSV and save path,label,score.")
    parser.add_argument("input_csv", type=Path, help="CSV with path,label columns.")
    parser.add_argument("--model", type=Path, required=True, help="ONNX model file.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV with path,label,score.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-bbox-crop", action="store_true")
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--on-missing", choices=["skip", "raise"], default="skip")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.model), providers=providers)
    print(f"Loaded {args.model}, providers: {session.get_providers()}")

    original_paths, resolved_paths, labels, bboxes = read_prediction_csv(
        args.input_csv, use_bbox_crop=args.use_bbox_crop,
    )
    original_paths, resolved_paths, labels, bboxes = filter_missing_files(
        original_paths, resolved_paths, labels, bboxes, args.on_missing,
    )

    scores = predict(session, resolved_paths, bboxes, args.batch_size, args.use_bbox_crop, args.margin)
    write_scores(args.output_csv, original_paths, labels, scores)
    print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
