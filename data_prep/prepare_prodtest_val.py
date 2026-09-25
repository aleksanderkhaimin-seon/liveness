#!/usr/bin/env python3
"""Build a liveness val CSV from ProdTest keys on the prod EFS mount.

s3_object_key looks like merchant/session/document_check/<id>. On this mount the
usable images are merchant/session/extracted-frames/*.jpeg (one row per frame).
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def session_dir(mount_root: Path, s3_object_key: str) -> Path:
    parts = Path(s3_object_key.strip().lstrip("/")).parts
    if len(parts) < 2:
        raise ValueError(f"Expected merchant/session/... key, got {s3_object_key!r}")
    return mount_root / parts[0] / parts[1]


def frame_paths(session: Path) -> list[Path]:
    frames_dir = session / "extracted-frames"
    if not frames_dir.is_dir():
        return []
    files = [
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=Path("data/ProdTest-0.2.csv"))
    parser.add_argument(
        "--mount-root",
        type=Path,
        default=Path(
            "/home/sagemaker-user/prod-data-efs/buckets/id-verification-internal-prod-eu-west-1-847433666304"
        ),
    )
    parser.add_argument("--output-csv", type=Path, default=Path("data/ProdTest-0.2-val.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)
    if not args.mount_root.is_dir():
        raise FileNotFoundError(args.mount_root)

    rows_in = 0
    missing_sessions = 0
    empty_frames = 0
    written = 0
    labels = Counter()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.input_csv.open("r", encoding="utf-8", newline="") as incoming, args.output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as outgoing:
        reader = csv.DictReader(incoming)
        if not reader.fieldnames or "s3_object_key" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{args.input_csv} must contain s3_object_key,label")
        writer = csv.DictWriter(outgoing, fieldnames=["path", "label", "dataset", "name"])
        writer.writeheader()

        for row in reader:
            rows_in += 1
            label = int(row["label"].strip())
            if label not in (0, 1):
                raise ValueError(f"Label must be 0 or 1, got {row['label']!r}")
            dataset = (row.get("mdsid") or "ProdTest").strip() or "ProdTest"
            session = session_dir(args.mount_root, row["s3_object_key"])
            if not session.is_dir():
                missing_sessions += 1
                continue
            frames = frame_paths(session)
            if not frames:
                empty_frames += 1
                continue
            for frame in frames:
                writer.writerow(
                    {
                        "path": str(frame),
                        "label": label,
                        "dataset": dataset,
                        "name": frame.name,
                    }
                )
                labels[label] += 1
                written += 1

    print(
        "\n".join(
            [
                f"input_rows: {rows_in}",
                f"missing_sessions: {missing_sessions}",
                f"sessions_without_frames: {empty_frames}",
                f"output_rows: {written}",
                f"label_0: {labels.get(0, 0)}",
                f"label_1: {labels.get(1, 0)}",
                f"output: {args.output_csv}",
            ]
        )
    )


if __name__ == "__main__":
    main()
