#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


EMPTY_VALUES = {"", "none", "null", "nan"}


def validate_bbox(value: str, csv_path: Path, row_number: int) -> str | None:
    raw = value.strip()
    if raw.lower() in EMPTY_VALUES:
        return None

    try:
        bbox = json.loads(raw)
    except json.JSONDecodeError as error:
        return f"{csv_path}:{row_number}: invalid JSON bbox {raw!r}: {error}"

    if not isinstance(bbox, list) or len(bbox) != 4:
        return f"{csv_path}:{row_number}: bbox must be [x1,y1,x2,y2], got {raw!r}"

    try:
        x1, y1, x2, y2 = [float(item) for item in bbox]
    except (TypeError, ValueError):
        return f"{csv_path}:{row_number}: bbox values must be numeric, got {raw!r}"

    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)):
        return f"{csv_path}:{row_number}: bbox values must be finite, got {raw!r}"
    if x2 <= x1 or y2 <= y1:
        return f"{csv_path}:{row_number}: bbox must satisfy x2 > x1 and y2 > y1, got {raw!r}"

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate path/label/bbox CSV before bbox-crop training.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--max-errors", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = []
    empty_count = 0
    valid_count = 0

    with args.csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise SystemExit(f"{args.csv_path}: missing header")

        missing_columns = {"path", "label", "bbox"} - set(reader.fieldnames)
        if missing_columns:
            raise SystemExit(f"{args.csv_path}: missing columns: {sorted(missing_columns)}")

        for row_number, row in enumerate(reader, start=2):
            bbox = row.get("bbox", "")
            error = validate_bbox(bbox, args.csv_path, row_number)
            if error:
                errors.append(error)
                if len(errors) >= args.max_errors:
                    break
            elif bbox.strip().lower() in EMPTY_VALUES:
                empty_count += 1
            else:
                valid_count += 1

    if errors:
        print("\n".join(errors))
        raise SystemExit(1)

    print(
        f"{args.csv_path}: bbox column OK. "
        f"valid={valid_count}, empty/full-image-fallback={empty_count}"
    )


if __name__ == "__main__":
    main()
