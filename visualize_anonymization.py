#!/usr/bin/env python3
"""Render every proposed anonymisation of a document frame side by side.

For each sampled image writes one sheet with the variants evaluated in
runs/train_reports (September 2026), and an index montage across all sampled
images:

    original               | doc only (crop, margin 0)   | mask:0.05 (+crop 25)  | mask:0 (+crop 25) | pixelate:64 (+crop 25)
    downscale:192          | downscale:128               | downscale_doc:96      | EXPORT doc:96 1:1 | patches 512 px

The first three rows reproduce the training-time --degrade transforms of
train_efficientnet_b2.py in PIL (same operations: area downsample, bilinear
upsample, per-channel-mean fill); the export panel is what
export_lowres_frames.py writes; patches come from sample_document_patches.py.

THE OUTPUT CONTAINS THE ORIGINAL DOCUMENTS. It is for internal review of the
anonymisation and stays inside the trusted zone like the manifests.

Example:

    python visualize_anonymization.py data/ProdTest-0.3_det_ik.csv --out-dir /secure/anon_examples --per-label 4
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_document_patches as sdp  # noqa: E402

EMPTY_VALUES = {"", "none", "null", "nan"}

# Production EER at 3 epochs from runs/train_reports (val = ProdTest-0.3), for the panel captions.
PROD_EER_3EP = {
    "original": 14.95,
    "doc_only": 24.81,
    "mask_005": 19.69,
    "mask_0": 22.95,
    "pixelate_64": 19.58,
    "downscale_192": 15.88,
    "downscale_128": 19.36,
    "downscale_doc_96": 16.20,
    "export_doc_96": 15.41,
    "patches": 37.95,
}
CAPTIONS = {
    "original": "original (full frame)",
    "doc_only": "doc only: crop, margin 0",
    "mask_005": "mask:0.05 + crop 25",
    "mask_0": "mask:0 + crop 25",
    "pixelate_64": "pixelate:64 + crop 25",
    "downscale_192": "downscale:192 (as model sees)",
    "downscale_128": "downscale:128 (as model sees)",
    "downscale_doc_96": "downscale_doc:96 (as model sees)",
    "export_doc_96": "EXPORT doc:96, 1:1 pixels",
    "patches": "patches 512 px (4 of 8)",
}
ORDER = ["original", "doc_only", "mask_005", "mask_0", "pixelate_64",
         "downscale_192", "downscale_128", "downscale_doc_96", "export_doc_96", "patches"]


# -- transforms (PIL mirrors of the tf ops in train_efficientnet_b2.py) -----

def crop_margin(image: Image.Image, bbox, margin_pct: float) -> Image.Image:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    r = margin_pct / 100.0
    box = (max(0, int(math.floor(x1 - w * r))), max(0, int(math.floor(y1 - h * r))),
           min(image.width, int(math.ceil(x2 + w * r))), min(image.height, int(math.ceil(y2 + h * r))))
    return image.crop(box)


def interior_rect(image: Image.Image, bbox, band: float):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return (max(0, int(math.floor(x1 + w * band))), max(0, int(math.floor(y1 + h * band))),
            min(image.width, int(math.ceil(x2 - w * band))), min(image.height, int(math.ceil(y2 - h * band))))


def mask_interior(image: Image.Image, bbox, band: float) -> Image.Image:
    rx1, ry1, rx2, ry2 = interior_rect(image, bbox, band)
    if rx2 <= rx1 or ry2 <= ry1:
        return image
    arr = np.asarray(image).astype(np.float32)
    fill = arr[ry1:ry2, rx1:rx2].mean(axis=(0, 1))
    arr[ry1:ry2, rx1:rx2] = fill
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def pixelate_interior(image: Image.Image, bbox, target_short: int) -> Image.Image:
    rx1, ry1, rx2, ry2 = interior_rect(image, bbox, 0.0)
    rw, rh = rx2 - rx1, ry2 - ry1
    short = min(rw, rh)
    if short <= target_short:
        return image
    factor = target_short / short
    region = image.crop((rx1, ry1, rx2, ry2))
    small = region.resize((max(1, round(rw * factor)), max(1, round(rh * factor))), Image.Resampling.BOX)
    back = small.resize((rw, rh), Image.Resampling.BILINEAR)
    out = image.copy()
    out.paste(back, (rx1, ry1))
    return out


def downscale_frame(image: Image.Image, long_side: float) -> Image.Image:
    factor = long_side / max(image.size)
    if factor >= 1.0:
        return image
    small = image.resize((max(1, round(image.width * factor)), max(1, round(image.height * factor))), Image.Resampling.BOX)
    return small.resize(image.size, Image.Resampling.BILINEAR)


def downscale_doc(image: Image.Image, bbox, doc_long: int) -> Image.Image:
    x1, y1, x2, y2 = bbox
    long = max(x2 - x1, y2 - y1)
    if long <= doc_long:
        return image
    return downscale_frame(image, max(image.size) * doc_long / long)


def export_doc(image: Image.Image, bbox, doc_long: int) -> Image.Image:
    """What export_lowres_frames.py --mode doc writes: one BOX resample, no upsample."""
    x1, y1, x2, y2 = bbox
    scale = min(1.0, doc_long / max(x2 - x1, y2 - y1))
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BOX)


def patch_tiles(row: dict, image: Image.Image, patch_size: int, seed: int) -> list[tuple[Image.Image, str]]:
    cfg = sdp.SamplerConfig(
        patch_size=patch_size, patches_per_image=8, margin=25.0,
        weights=(("interior", 0.5), ("edge", 0.5), ("exterior", 0.0)),
        edge_jitter=0.2, min_zone_frac=0.15, min_center_distance=0.25, max_tries=40, backfill=True,
        max_exclusion_overlap=0.0, min_std=0.0, clip_to_margin=False, image_format="png", quality=100,
        seed=seed, dry_run=True,
    )
    records, _, error = sdp.process_row(
        {"source_path": row["source_path"], "label": row["label"], "bbox": row["bbox"], "exclusions": []},
        cfg, None, 0, None,
    )
    if error or not records:
        return []
    tiles = []
    for zone in ("interior", "edge"):
        for rec in [r for r in records if r["zone"] == zone][:2]:
            tiles.append((image.crop((rec["x"], rec["y"], rec["x"] + rec["size"], rec["y"] + rec["size"])), zone))
    return tiles


# -- rendering --------------------------------------------------------------

def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def fit(image: Image.Image, box: int, bg=(22, 22, 26), resample=Image.Resampling.BILINEAR) -> Image.Image:
    """Letterbox `image` into a box x box tile without upscaling past 1:1."""
    scale = min(box / image.width, box / image.height, 1.0)
    im = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), resample) if scale < 1.0 else image
    tile = Image.new("RGB", (box, box), bg)
    tile.paste(im, ((box - im.width) // 2, (box - im.height) // 2))
    return tile


def one_to_one(image: Image.Image, box: int, bg=(22, 22, 26)) -> Image.Image:
    """Place an image at exactly its own pixel size inside the tile (export panel)."""
    tile = Image.new("RGB", (box, box), bg)
    im = image if max(image.size) <= box else fit(image, box, bg)
    tile.paste(im, ((box - im.width) // 2, (box - im.height) // 2))
    return tile


def patches_panel(tiles: list[tuple[Image.Image, str]], box: int, bg=(22, 22, 26)) -> Image.Image:
    panel = Image.new("RGB", (box, box), bg)
    if not tiles:
        return panel
    half = box // 2 - 2
    draw = ImageDraw.Draw(panel)
    for i, (tile, zone) in enumerate(tiles[:4]):
        t = tile.resize((half, half), Image.Resampling.BILINEAR)
        x, y = (i % 2) * (half + 4), (i // 2) * (half + 4)
        panel.paste(t, (x, y))
        draw.rectangle((x, y, x + half - 1, y + half - 1), outline=sdp.ZONE_COLOURS[zone], width=2)
    return panel


def variants(row: dict, image: Image.Image, patch_size: int, seed: int) -> dict[str, Image.Image | list]:
    bbox = row["bbox"]
    return {
        "original": image,
        "doc_only": crop_margin(image, bbox, 0.0),
        "mask_005": crop_margin(mask_interior(image, bbox, 0.05), bbox, 25.0),
        "mask_0": crop_margin(mask_interior(image, bbox, 0.0), bbox, 25.0),
        "pixelate_64": crop_margin(pixelate_interior(image, bbox, 64), bbox, 25.0),
        "downscale_192": downscale_frame(image, 192),
        "downscale_128": downscale_frame(image, 128),
        "downscale_doc_96": downscale_doc(image, bbox, 96),
        "export_doc_96": export_doc(image, bbox, 96),
        "patches": patch_tiles(row, image, patch_size, seed),
    }


def render_sheet(row: dict, image: Image.Image, vs: dict, box: int, eer_labels: bool) -> Image.Image:
    cols, rows_n, pad, cap_h = 5, 2, 10, 34
    sheet = Image.new("RGB", (cols * (box + pad) + pad, rows_n * (box + cap_h + pad) + pad + 28), (22, 22, 26))
    draw = ImageDraw.Draw(sheet)
    f_cap, f_small, f_head = font(14), font(11), font(15)
    label_txt = "attack / replay" if row["label"] == 1 else "bona fide"
    draw.text((pad, 6), f"label {row['label']} ({label_txt})   {image.width}x{image.height}   bbox {json.dumps([int(v) for v in row['bbox']])}", fill=(235, 235, 235), font=f_head)
    for i, key in enumerate(ORDER):
        x = pad + (i % cols) * (box + pad)
        y = 28 + pad + (i // cols) * (box + cap_h + pad)
        if key == "patches":
            tile = patches_panel(vs[key], box)
        elif key == "export_doc_96":
            tile = one_to_one(vs[key], box)
        else:
            tile = fit(vs[key], box)
        sheet.paste(tile, (x, y))
        caption = CAPTIONS[key]
        if key == "export_doc_96":
            caption += f"  ({vs[key].width}x{vs[key].height})"
        draw.text((x, y + box + 4), caption, fill=(235, 235, 235), font=f_cap)
        if eer_labels and key in PROD_EER_3EP:
            draw.text((x, y + box + 20), f"prod EER {PROD_EER_3EP[key]:.1f}% (3 ep)", fill=(255, 190, 90), font=f_small)
    return sheet


# -- main ---------------------------------------------------------------------

def read_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row_number, row in enumerate(reader, start=2):
            raw = (row.get("bbox") or "").strip()
            if raw.lower() in EMPTY_VALUES:
                continue
            try:
                bbox = tuple(float(v) for v in json.loads(raw))
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            source = Path(row["path"].strip())
            if not source.is_absolute():
                source = csv_path.parent / source
            rows.append({"source_path": str(source), "label": int(row["label"]), "bbox": bbox,
                         "dataset": row.get("dataset") or row.get("predicted_class") or ""})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=3, help="Images sampled per label.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tile", type=int, default=320, help="Panel size in px on the per-image sheets.")
    parser.add_argument("--index-tile", type=int, default=128, help="Panel size on the index montage.")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--no-eer-labels", action="store_true", help="Omit the production-EER captions.")
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    if not rows:
        raise SystemExit(f"{args.input_csv}: no rows with a usable bbox")
    rng = random.Random(args.seed)
    chosen = []
    for label in (0, 1):
        pool = [r for r in rows if r["label"] == label]
        rng.shuffle(pool)
        chosen.extend(pool[: args.per_label])
    if not chosen:
        raise SystemExit("Nothing to render")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "README.txt").write_text(
        "These sheets contain the ORIGINAL documents next to their anonymised variants.\n"
        "Internal review material only; keep inside the trusted zone with the manifests.\n", encoding="utf-8")

    index_rows = []
    for n, row in enumerate(chosen):
        try:
            image = Image.open(row["source_path"]).convert("RGB")
        except Exception as error:
            print(f"skip {row['source_path']}: {error}", file=sys.stderr)
            continue
        vs = variants(row, image, args.patch_size, args.seed)
        sheet = render_sheet(row, image, vs, args.tile, not args.no_eer_labels)
        tag = hashlib.blake2b(row["source_path"].encode("utf-8"), digest_size=4).hexdigest()
        name = f"label{row['label']}_{n:02d}_{tag}.png"
        sheet.save(args.out_dir / name, optimize=True)
        print(f"  {name}  <- {row['source_path']}")
        index_rows.append([
            patches_panel(vs[k], args.index_tile) if k == "patches"
            else one_to_one(vs[k], args.index_tile) if k == "export_doc_96"
            else fit(vs[k], args.index_tile)
            for k in ORDER
        ] + [row["label"]])

    if index_rows:
        t, pad, head = args.index_tile, 6, 22
        index = Image.new("RGB", (len(ORDER) * (t + pad) + pad + 70, len(index_rows) * (t + pad) + pad + head), (22, 22, 26))
        draw = ImageDraw.Draw(index)
        f = font(10)
        for j, key in enumerate(ORDER):
            draw.text((70 + pad + j * (t + pad), 4), CAPTIONS[key][:22], fill=(235, 235, 235), font=f)
        for i, tiles in enumerate(index_rows):
            y = head + pad + i * (t + pad)
            draw.text((pad, y + t // 2 - 6), f"label {tiles[-1]}", fill=(235, 235, 235), font=f)
            for j, tile in enumerate(tiles[:-1]):
                index.paste(tile, (70 + pad + j * (t + pad), y))
        index.save(args.out_dir / "index.png", optimize=True)
        print(f"  index.png  ({len(index_rows)} images x {len(ORDER)} variants)")

    print(f"\nKEEP BEHIND: {args.out_dir}  (contains original documents)")


if __name__ == "__main__":
    main()
