#!/usr/bin/env python3
"""Turn document-detector CSVs into path,label,bbox input for synth_screen_replay.py.

The detector file stores a quadrilateral (corner_tl/tr/br/bl). The replay script
wants an axis-aligned box as JSON [x1,y1,x2,y2]. Label is written for the CSV
shape; synth_screen_replay.py ignores it and uses its own --label.

Examples:

    python data_prep/docdet_to_replay_csv.py data/PS-generated-batch-1_docdet_ik.csv
    python data_prep/docdet_to_replay_csv.py data/*_docdet_ik.csv --accept-only
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CORNER_X = ("corner_tl_x", "corner_tr_x", "corner_br_x", "corner_bl_x")
CORNER_Y = ("corner_tl_y", "corner_tr_y", "corner_br_y", "corner_bl_y")
REQUIRED = ("path", "width", "height", *CORNER_X, *CORNER_Y)
MIN_SIDE = 8


def default_output(source: Path) -> Path:
    name = source.name
    if name.endswith("_docdet_ik.csv"):
        name = name[: -len("_docdet_ik.csv")] + "_replay.csv"
    elif name.endswith(".csv"):
        name = name[: -len(".csv")] + "_replay.csv"
    else:
        name = name + "_replay.csv"
    return source.with_name(name)


def image_path(row: dict[str, str]) -> str:
    path = (row.get("path") or "").strip()
    if path:
        return path
    folder = (row.get("file_path") or "").strip()
    name = (row.get("file_name") or "").strip()
    if folder and name:
        return str(Path(folder) / name)
    return ""


def axis_aligned_box(row: dict[str, str], min_side: int) -> tuple[int, int, int, int] | None:
    xs = [float(row[key]) for key in CORNER_X]
    ys = [float(row[key]) for key in CORNER_Y]
    width = float(row["width"])
    height = float(row["height"])
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    ix1, iy1, ix2, iy2 = (int(round(value)) for value in (x1, y1, x2, y2))
    if ix2 - ix1 < min_side or iy2 - iy1 < min_side:
        return None
    return ix1, iy1, ix2, iy2


def convert(source: Path, dest: Path, label: int, accept_only: bool, min_side: int) -> tuple[int, int]:
    written = 0
    skipped = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig", newline="") as incoming, dest.open(
        "w", encoding="utf-8", newline=""
    ) as outgoing:
        reader = csv.DictReader(incoming)
        missing = [name for name in REQUIRED if not reader.fieldnames or name not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{source} is missing columns: {', '.join(missing)}")
        writer = csv.DictWriter(outgoing, fieldnames=["path", "label", "bbox"])
        writer.writeheader()
        for row in reader:
            if accept_only and (row.get("decision") or "").strip().upper() != "ACCEPT":
                skipped += 1
                continue
            path = image_path(row)
            if not path:
                skipped += 1
                continue
            try:
                box = axis_aligned_box(row, min_side)
            except ValueError:
                skipped += 1
                continue
            if box is None:
                skipped += 1
                continue
            writer.writerow({
                "path": path,
                "label": str(label),
                "bbox": json.dumps(list(box), separators=(",", ":")),
            })
            written += 1
    return written, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", type=Path, nargs="+", help="Detector CSV(s) with corner_* columns.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV for a single input, or a directory when several inputs are given. "
        "Default: <name>_replay.csv next to each input (_docdet_ik is replaced).",
    )
    parser.add_argument("--label", type=int, choices=(0, 1), default=1, help="Label column (default 1, live).")
    parser.add_argument("--accept-only", action="store_true", help="Keep rows whose decision is ACCEPT.")
    parser.add_argument("--min-side", type=int, default=MIN_SIDE, help="Drop boxes smaller than this, in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is not None and len(args.sources) > 1 and not args.output.is_dir():
        raise SystemExit("--output must be a directory when converting more than one CSV")
    for source in args.sources:
        if not source.is_file():
            raise SystemExit(f"not a file: {source}")
        if args.output is None:
            dest = default_output(source)
        elif len(args.sources) == 1 and not args.output.is_dir():
            dest = args.output
        else:
            dest = args.output / default_output(source).name
        if dest.resolve() == source.resolve():
            raise SystemExit(f"refusing to overwrite the detector CSV: {source}")
        written, skipped = convert(source, dest, args.label, args.accept_only, args.min_side)
        print(f"{source} -> {dest}: wrote {written}, skipped {skipped}")


if __name__ == "__main__":
    main()
