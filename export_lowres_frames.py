#!/usr/bin/env python3
"""Export low-resolution frames as an anonymised training representation.

Rationale (runs/train_reports, Sept 2026): a model trained on full frames
downscaled to a long side of 192 px matched the un-degraded baseline on both
the Pinterest test set (3.0 vs 3.2% EER) and production validation (15.9 vs
15.0%), while native-resolution patches fell to chance on production. The
signal that transfers is coarse; the PII is fine. So the export is simply the
frame at a resolution where the document is ~90 px wide.

Two scaling modes:

    --mode frame --size 192   frame long side -> 192 px (what the training run did)
    --mode doc   --size 96    frame scaled so the DOCUMENT long side -> 96 px;
                              equal to frame:192 for the median capture, harsher
                              for close-ups, so illegibility holds per image

Outputs in --out-dir:

    frames.csv    path,label,bbox    bbox rescaled to the export -- safe to move
    manifest.csv  id -> source_path, scale, original size -- KEEP BEHIND
    summary.json  class balance, skip reasons per label, legibility proxy at
                  the export scale (text px, face px per image)

Images are resampled once with a box filter (area average, matching
tf.image.resize(method="area")) and written as PNG. The training pipeline
then does a single bilinear resize to IMAGE_SIZE, versus two in the
--degrade downscale run; a confirmation job on the exported files closes
that gap empirically.

Example:

    python export_lowres_frames.py data/train.csv --out-dir /data/anon/train --mode doc --size 96 --workers 16
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

EMPTY_VALUES = {"", "none", "null", "nan"}
TEXT_FRAC = 0.04   # field text height as a fraction of the document short side
FACE_FRAC = 0.35   # face height as a fraction of the document short side
TEXT_LEGIBLE_PX = 5.0
FACE_MATCHABLE_PX = 40.0


def parse_bbox(raw: str, csv_path: Path, row_number: int) -> tuple[float, float, float, float] | None:
    text = (raw or "").strip()
    if text.lower() in EMPTY_VALUES:
        return None
    try:
        values = json.loads(text)
        x1, y1, x2, y2 = [float(v) for v in values]
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid bbox at {csv_path}:{row_number}: {text!r}") from error
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate bbox at {csv_path}:{row_number}: {text!r}")
    return (x1, y1, x2, y2)


def read_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or {"path", "label"} - set(reader.fieldnames):
            raise ValueError(f"{csv_path} must contain path,label columns")
        for row_number, row in enumerate(reader, start=2):
            raw_path = row["path"].strip()
            if not raw_path:
                raise ValueError(f"Empty path at {csv_path}:{row_number}")
            source = Path(raw_path)
            if not source.is_absolute():
                source = csv_path.parent / source
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Label must be 0 or 1 at {csv_path}:{row_number}")
            rows.append({"source_path": str(source), "label": label,
                         "bbox": parse_bbox(row.get("bbox", ""), csv_path, row_number)})
    return rows


@dataclass(frozen=True)
class ExportConfig:
    mode: str
    size: int
    seed: int
    dry_run: bool


def export_row(row: dict, cfg: ExportConfig, out_dir: str | None, row_index: int) -> tuple[dict | None, Counter, str | None]:
    stats: Counter = Counter()
    label = row["label"]

    def skipped(reason: str, error: str | None = None):
        return (None, Counter({reason: 1, f"{reason}__label_{label}": 1}), error)

    try:
        image = Image.open(row["source_path"])
        width, height = image.size
    except Exception as error:
        return skipped("rows_failed", f"{row['source_path']}: {error}")

    bbox = row["bbox"]
    if cfg.mode == "doc" and bbox is None:
        return skipped("rows_no_bbox")

    if cfg.mode == "frame":
        scale = min(1.0, cfg.size / max(width, height))
    else:
        x1, y1, x2, y2 = bbox
        doc_long = max(x2 - x1, y2 - y1)
        scale = min(1.0, cfg.size / max(doc_long, 1.0))
    if scale >= 1.0:
        stats["rows_not_downscaled"] += 1  # already at or below the target; exported unchanged

    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))

    digest = hashlib.blake2b(f"{cfg.seed}|{row_index}|{row['source_path']}".encode("utf-8"), digest_size=16)
    frame_id = digest.hexdigest()
    relative = f"frames/{label}/{frame_id[:2]}/{frame_id}.png"

    scaled_bbox = None
    if bbox is not None:
        scaled_bbox = [round(v * scale, 2) for v in bbox]
        doc_short = min(bbox[2] - bbox[0], bbox[3] - bbox[1]) * scale
        text_px = TEXT_FRAC * doc_short
        face_px = FACE_FRAC * doc_short
    else:
        text_px = face_px = float("nan")

    if not cfg.dry_run and out_dir:
        try:
            image = image.convert("RGB")
            if scale < 1.0:
                image = image.resize((new_w, new_h), Image.Resampling.BOX)
            target = Path(out_dir) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, format="PNG", compress_level=6)
        except Exception as error:
            return skipped("rows_failed", f"{row['source_path']}: {error}")
    image.close()

    stats["rows_ok"] += 1
    record = {
        "frame_id": frame_id,
        "relative_path": relative,
        "label": label,
        "bbox": json.dumps(scaled_bbox, separators=(",", ":")) if scaled_bbox else "",
        "source_path": row["source_path"],
        "scale": round(scale, 6),
        "source_width": width,
        "source_height": height,
        "export_width": new_w,
        "export_height": new_h,
        "text_px": round(text_px, 2) if not math.isnan(text_px) else "",
        "face_px": round(face_px, 2) if not math.isnan(face_px) else "",
    }
    return (record, stats, None)


def legibility(records: list[dict]) -> dict:
    out = {}
    for label in sorted({r["label"] for r in records}):
        text = np.array([r["text_px"] for r in records if r["label"] == label and r["text_px"] != ""], dtype=float)
        face = np.array([r["face_px"] for r in records if r["label"] == label and r["face_px"] != ""], dtype=float)
        out[str(label)] = {
            "n_with_bbox": int(len(text)),
            "text_px_p50": round(float(np.median(text)), 2) if len(text) else None,
            "text_px_p90": round(float(np.percentile(text, 90)), 2) if len(text) else None,
            "share_text_illegible": round(float(np.mean(text < TEXT_LEGIBLE_PX)), 4) if len(text) else None,
            "face_px_p50": round(float(np.median(face)), 2) if len(face) else None,
            "face_px_p90": round(float(np.percentile(face, 90)), 2) if len(face) else None,
            "share_face_unmatchable": round(float(np.mean(face < FACE_MATCHABLE_PX)), 4) if len(face) else None,
        }
    return out


def problems_for(rows: list[dict], records: list[dict], stats: Counter) -> list[str]:
    problems = []
    rows_in = Counter(r["label"] for r in rows)
    out = Counter(r["label"] for r in records)
    if len(rows_in) >= 2 and len(out) < 2:
        problems.append(f"FATAL: output contains only label {sorted(out)}; check skips_by_label")
    elif not out:
        problems.append("FATAL: no frames exported")
    for reason in sorted({k.partition("__label_")[0] for k in stats if "__label_" in k}):
        rates = {l: stats.get(f"{reason}__label_{l}", 0) / n for l, n in rows_in.items() if n}
        if len(rates) >= 2 and max(rates.values()) - min(rates.values()) > 0.20:
            problems.append(f"WARNING: {reason} is label-correlated: " + ", ".join(f"label {l}: {r:.0%}" for l, r in sorted(rates.items())))
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("frame", "doc"), default="doc",
                        help="frame: frame long side -> --size. doc: document long side -> --size (needs bbox).")
    parser.add_argument("--size", type=int, default=96, help="Target long side in px (frame:192 ~ doc:96 for the median capture).")
    parser.add_argument("--seed", type=int, default=0, help="Seeds the opaque ids and the output shuffle.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Compute ids, scales and legibility without writing images.")
    parser.add_argument("--allow-single-class", action="store_true")
    parser.add_argument(
        "--csv-name",
        default="frames.csv",
        help="Name of the output CSV. Give train/validation/test distinct names: sagemaker_job/launch.py "
        "stages every config CSV as data/<basename>, so three files called frames.csv overwrite each other.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute image paths into frames.csv (resolved against --out-dir). Required for "
        "sagemaker_job/launch.py, which stages the CSV away from the images and rewrites paths by EFS prefix.",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for csv_path in args.input_csv:
        rows.extend(read_rows(csv_path))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No rows to export")

    cfg = ExportConfig(mode=args.mode, size=args.size, seed=args.seed, dry_run=args.dry_run)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rows: {len(rows)}  mode={cfg.mode}  size={cfg.size}px  dry_run={cfg.dry_run}")

    records: list[dict] = []
    stats: Counter = Counter()
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(export_row, row, cfg, None if cfg.dry_run else str(args.out_dir), i)
                   for i, row in enumerate(rows)]
        for done, future in enumerate(as_completed(futures), start=1):
            record, row_stats, error = future.result()
            stats.update(row_stats)
            if record:
                records.append(record)
            if error:
                errors.append(error)
            if done % 1000 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)}", file=sys.stderr)

    random.Random(args.seed).shuffle(records)
    problems = problems_for(rows, records, stats)

    summary = {
        "rows_total": len(rows),
        "rows_by_label": {str(k): v for k, v in sorted(Counter(r["label"] for r in rows).items())},
        "frames_total": len(records),
        "by_label": {str(k): v for k, v in sorted(Counter(r["label"] for r in records).items())},
        "skips_by_label": {
            label: {k.partition("__label_")[0]: v for k, v in stats.items() if k.endswith(f"__label_{label}")}
            for label in sorted({k.partition("__label_")[2] for k in stats if "__label_" in k})
        },
        "problems": problems,
        "legibility_at_export_scale": legibility(records),
        "export_long_side_p50": int(np.median([max(r["export_width"], r["export_height"]) for r in records])) if records else None,
        "counters": dict(sorted(stats.items())),
        "errors": len(errors),
        "config": {"mode": cfg.mode, "size": cfg.size, "seed": cfg.seed, "inputs": [str(p) for p in args.input_csv]},
    }

    if not cfg.dry_run:
        with (args.out_dir / args.csv_name).open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["path", "label", "bbox"])
            root = args.out_dir.resolve()
            for r in records:
                path = str(root / r["relative_path"]) if args.absolute_paths else r["relative_path"]
                writer.writerow([path, r["label"], r["bbox"]])
        fields = ["frame_id", "relative_path", "label", "source_path", "scale", "source_width", "source_height",
                  "export_width", "export_height", "text_px", "face_px"]
        with (args.out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if errors:
        print(f"\n{len(errors)} unreadable images, first 10:", file=sys.stderr)
        for e in errors[:10]:
            print(f"  {e}", file=sys.stderr)
    if not cfg.dry_run:
        print(f"\nSafe to move : {args.out_dir / args.csv_name} + frames/")
        print(f"KEEP BEHIND  : {args.out_dir / 'manifest.csv'}")
    if problems:
        print("\nPROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        if any(p.startswith("FATAL") for p in problems) and not args.allow_single_class:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
