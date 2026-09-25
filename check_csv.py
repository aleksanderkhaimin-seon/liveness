#!/usr/bin/env python3
"""Check image files listed in a CSV for existence and readability by TensorFlow."""
import argparse
import csv
import sys
from pathlib import Path

import tensorflow as tf


def check_csv(csv_path: Path, output_csv: Path | None, max_errors: int) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain a 'path' column")
        fieldnames = reader.fieldnames
        rows = list(reader)

    total = len(rows)
    missing = []
    unreadable = []
    good_rows = []

    for i, row in enumerate(rows, start=1):
        raw_path = row["path"].strip()
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = csv_path.parent / image_path

        print(f"\r{i}/{total}", end="", flush=True)

        if not image_path.exists():
            missing.append(raw_path)
            if max_errors and len(missing) + len(unreadable) >= max_errors:
                break
            continue

        try:
            raw = tf.io.read_file(str(image_path))
            tf.io.decode_image(raw, channels=3, expand_animations=False)
            good_rows.append(row)
        except Exception as e:
            unreadable.append((raw_path, str(e)))
            if max_errors and len(missing) + len(unreadable) >= max_errors:
                break

    print()

    ok = len(good_rows)
    print(f"\nTotal:     {total}")
    print(f"OK:        {ok}")
    print(f"Missing:   {len(missing)}")
    print(f"Unreadable:{len(unreadable)}")

    if missing:
        print("\n--- Missing files ---")
        for p in missing:
            print(f"  {p}")

    if unreadable:
        print("\n--- Unreadable files ---")
        for p, err in unreadable:
            print(f"  {p}")
            print(f"    {err}")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(good_rows)
        print(f"\nClean CSV saved to: {output_csv}")

    if missing or unreadable:
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check image files in a CSV for TF readability.")
    parser.add_argument("csv", type=Path, help="CSV with a 'path' column.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Save clean rows (readable files only) to this CSV.")
    parser.add_argument("--max-errors", type=int, default=0, help="Stop after this many errors (0 = check all).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_csv(args.csv, args.output_csv, args.max_errors)
