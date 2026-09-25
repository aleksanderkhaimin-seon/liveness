#!/usr/bin/env python3
"""Estimate PII legibility vs export resolution from document geometry alone.

No OCR or face detector needed: the CSV carries the frame size and the
document bbox, and ID layouts are standardised enough for two proxies:

    field text height  ~ 4%  of the document's short side  (ID-1 cards ~2.2 mm / 54 mm)
    face height        ~ 35% of the document's short side  (portrait ~50% of height,
                                                             face ~70% of portrait)

Legibility thresholds (conservative, i.e. biased toward calling things legible):

    text  : cap height >= 5 px is potentially readable by OCR
    face  : height >= 40 px is matchable by common face-recognition models;
            below 24 px recognition is negligible

For each candidate export resolution (long side, aspect preserved) the script
reports the share of images whose text / face would fall below those
thresholds, per label and per source dataset, plus how often the document
fills the frame (no edge/exterior zone) and how confident the corner detector
was (for quad-based interior masking).

Example:

    python legibility_proxy.py data/train_....csv --resolutions 512,384,256,192,128,96
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TEXT_FRAC = 0.04
FACE_FRAC = 0.35
TEXT_LEGIBLE_PX = 5.0
FACE_MATCHABLE_PX = 40.0
FACE_NEGLIGIBLE_PX = 24.0
FILLS_FRAME_TOL = 0.02  # bbox within 2% of frame size on both axes


def load(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        need = {"label", "bbox", "width", "height"}
        missing = need - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{csv_path}: missing columns {sorted(missing)}")
        for row in reader:
            try:
                x1, y1, x2, y2 = [float(v) for v in json.loads(row["bbox"])]
                width, height = float(row["width"]), float(row["height"])
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            if width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
                continue
            corner_conf = None
            peaks = [row.get(f"peak_{c}", "") for c in ("tl", "tr", "br", "bl")]
            if all(p not in ("", None) for p in peaks):
                corner_conf = min(float(p) for p in peaks)
            rows.append(
                {
                    "label": int(row["label"]),
                    "dataset": row.get("dataset", "?"),
                    "doc_type": row.get("predicted_class", "?"),
                    "frame_long": max(width, height),
                    "doc_w": x2 - x1,
                    "doc_h": y2 - y1,
                    "fills_frame": (x2 - x1) >= width * (1 - FILLS_FRAME_TOL)
                    and (y2 - y1) >= height * (1 - FILLS_FRAME_TOL),
                    "corner_conf": corner_conf,
                }
            )
    return rows


def table(rows: list[dict], resolutions: list[int], key: str) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)

    header = f"{key:<28} {'n':>6} {'doc/frame':>9} {'fills':>6} | " + " ".join(
        f"{'text<5px@' + str(r):>13}" for r in resolutions
    ) + " | " + " ".join(f"{'face<40@' + str(r):>11}" for r in resolutions)
    print(header)
    print("-" * len(header))
    for name, group in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        doc_short = np.array([min(r["doc_w"], r["doc_h"]) for r in group])
        doc_long = np.array([max(r["doc_w"], r["doc_h"]) for r in group])
        frame_long = np.array([r["frame_long"] for r in group])
        fills = np.mean([r["fills_frame"] for r in group])
        doc_frac = float(np.median(doc_long / frame_long))

        text_cells, face_cells = [], []
        for res in resolutions:
            scale = np.minimum(1.0, res / frame_long)  # never upscale
            text_px = TEXT_FRAC * doc_short * scale
            face_px = FACE_FRAC * doc_short * scale
            text_cells.append(f"{np.mean(text_px < TEXT_LEGIBLE_PX):>12.0%} ")
            face_cells.append(f"{np.mean(face_px < FACE_MATCHABLE_PX):>10.0%} ")
        print(
            f"{name[:28]:<28} {len(group):>6} {doc_frac:>9.2f} {fills:>6.0%} | "
            + " ".join(text_cells) + " | " + " ".join(face_cells)
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--resolutions", default="512,384,256,192,128,96",
                        help="Export long-side resolutions to evaluate.")
    args = parser.parse_args()
    resolutions = [int(v) for v in args.resolutions.split(",")]

    rows = load(args.csv_path)
    print(f"{args.csv_path.name}: {len(rows)} rows with usable geometry\n")

    frame_long = np.array([r["frame_long"] for r in rows])
    doc_short = np.array([min(r["doc_w"], r["doc_h"]) for r in rows])
    print("Frame long side   p10/p50/p90 :", *[f"{v:.0f}" for v in np.percentile(frame_long, [10, 50, 90])])
    print("Doc short side    p10/p50/p90 :", *[f"{v:.0f}" for v in np.percentile(doc_short, [10, 50, 90])])
    print(f"Native text height (~{TEXT_FRAC:.0%} of doc short side) p50: {np.median(TEXT_FRAC * doc_short):.0f} px")
    print(f"Native face height (~{FACE_FRAC:.0%} of doc short side) p50: {np.median(FACE_FRAC * doc_short):.0f} px")
    confs = [r["corner_conf"] for r in rows if r["corner_conf"] is not None]
    if confs:
        confs = np.array(confs)
        print(f"Corner detector min-peak      p10/p50: {np.percentile(confs, 10):.2f} / {np.median(confs):.2f}; "
              f"rows with a corner below 0.3: {np.mean(confs < 0.3):.1%}")
    print()

    print("Columns: doc/frame = median document long side / frame long side; fills = document "
          "covers the frame (no edge/exterior zone);\n text<5px@R = share of images whose field "
          "text is under 5 px at long side R; face<40@R = share whose face is under 40 px.\n")
    table(rows, resolutions, "label")
    table(rows, resolutions, "dataset")
    table(rows, resolutions, "doc_type")


if __name__ == "__main__":
    main()
