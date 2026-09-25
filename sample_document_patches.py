#!/usr/bin/env python3
"""Sample anonymising texture patches from bbox-cropped document images.

The document PAD signal (moire, screen bezel, print halftone, glare falloff,
sensor noise) is spread over the whole document surface and, importantly, over
the document/background boundary. The identity content (portrait, MRZ, number,
name, DOB) is localised. Sampling small patches therefore keeps almost all of
the label-carrying signal while destroying the readable content of any single
sample.

Zones, relative to the document bbox B and the margin-expanded region R:

    interior  patch lies fully inside B           surface texture, moire, halftone
    edge      patch straddles the boundary of B   bezel, doc-to-background transition
    exterior  patch lies inside R but outside B   screen frame, hand, desk, glare

Only `exterior` is bounded by the margin. `edge` patches are centred on the
boundary of B and are bounded by the image, so they always capture both sides
even when the margin band is narrower than half a patch.

Two CSVs are written:

    patches.csv   path,label,zone            -- safe to move out of the trusted zone
    manifest.csv  patch_id,source_path,...   -- the re-identification map, keep it behind

Patches are emitted at native resolution with no resampling, losslessly as PNG
by default. Downscaling aliases moire away and JPEG re-encoding overwrites the
compression fingerprint, so either would remove the signal this dataset exists
to carry. The default patch size matches IMAGE_SIZE in train_efficientnet_b2.py
so that the bilinear resize in the input pipeline is an identity op.

This script is a sampler, not a PII gate. Pass field-level detections (portrait,
MRZ, signature) through --exclude-column so overlapping patches are rejected.

Example:

    python sample_document_patches.py data/train_Seon-DocX-exp13-bbox.csv \\
        --out-dir data/patches_exp13 \\
        --patch-size 512 --patches-per-image 8 --margin 25 \\
        --zone-weights interior=0.5,edge=0.5,exterior=0.0 \\
        --overlay-dir /tmp/patch_overlays --overlay-limit 25
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
from PIL import Image, ImageDraw

EMPTY_VALUES = {"", "none", "null", "nan"}
ZONES = ("interior", "edge", "exterior")
ZONE_COLOURS = {"interior": (0, 200, 255), "edge": (255, 60, 60), "exterior": (255, 220, 0)}

Box = tuple[int, int, int, int]


# -- CSV helpers (mirrors predict_onnx_csv.py) -------------------------------

def normalize_bbox_value(raw_bbox: str, csv_path: Path, row_number: int) -> Box | None:
    bbox = raw_bbox.strip()
    if bbox.lower() in EMPTY_VALUES:
        return None
    try:
        values = json.loads(bbox)
    except json.JSONDecodeError as error:
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
    return (x1, y1, x2, y2)


def normalize_exclusions(raw: str, csv_path: Path, row_number: int) -> list[Box]:
    """Parse a JSON list of [x1,y1,x2,y2] regions that patches must not touch."""
    text = raw.strip()
    if text.lower() in EMPTY_VALUES:
        return []
    try:
        values = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid exclusion JSON at {csv_path}:{row_number}: {text!r}") from error
    if not isinstance(values, list):
        raise ValueError(f"Exclusions must be a JSON list at {csv_path}:{row_number}: {text!r}")
    if values and not isinstance(values[0], list):
        values = [values]
    boxes = []
    for box in values:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"Exclusion must be [x1,y1,x2,y2] at {csv_path}:{row_number}: {box!r}")
        x1, y1, x2, y2 = [float(v) for v in box]
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def read_input_csv(csv_path: Path, exclude_column: str | None) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} has no header")
        missing = {"path", "label", "bbox"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{csv_path}: missing columns: {sorted(missing)}")
        if exclude_column and exclude_column not in reader.fieldnames:
            raise ValueError(f"{csv_path}: missing exclusion column {exclude_column!r}")

        for row_number, row in enumerate(reader, start=2):
            raw_path = row["path"].strip()
            if not raw_path:
                raise ValueError(f"Empty image path at {csv_path}:{row_number}")
            source = Path(raw_path)
            if not source.is_absolute():
                source = csv_path.parent / source
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Label must be 0 or 1 at {csv_path}:{row_number}")
            rows.append(
                {
                    "source_path": str(source),
                    "label": label,
                    "bbox": normalize_bbox_value(row.get("bbox", ""), csv_path, row_number),
                    "exclusions": normalize_exclusions(row.get(exclude_column, ""), csv_path, row_number)
                    if exclude_column
                    else [],
                }
            )
    return rows


# -- geometry ----------------------------------------------------------------

def clip_box(box: Box, width: int, height: int) -> Box | None:
    x1 = max(0, int(math.floor(box[0])))
    y1 = max(0, int(math.floor(box[1])))
    x2 = min(width, int(math.ceil(box[2])))
    y2 = min(height, int(math.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def expand_bbox(bbox: Box, margin: float, width: int, height: int) -> Box | None:
    """Margin is a percentage of the box dimensions, as in crop_by_bbox()."""
    x1, y1, x2, y2 = bbox
    ratio = margin / 100.0
    box_width = x2 - x1
    box_height = y2 - y1
    return clip_box(
        (
            x1 - box_width * ratio,
            y1 - box_height * ratio,
            x2 + box_width * ratio,
            y2 + box_height * ratio,
        ),
        width,
        height,
    )


def intersect_area(a: Box, b: Box) -> int:
    overlap_w = min(a[2], b[2]) - max(a[0], b[0])
    overlap_h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, overlap_w) * max(0, overlap_h)


def perimeter_point(rng: random.Random, box: Box) -> tuple[float, float, float, float]:
    """A uniform point on the perimeter of box, with its outward unit normal."""
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    offset = rng.uniform(0.0, 2.0 * (box_width + box_height))
    if offset < box_width:
        return (x1 + offset, y1, 0.0, -1.0)
    offset -= box_width
    if offset < box_height:
        return (x2, y1 + offset, 1.0, 0.0)
    offset -= box_height
    if offset < box_width:
        return (x2 - offset, y2, 0.0, 1.0)
    offset -= box_width
    return (x1, y2 - offset, -1.0, 0.0)


def place(x: float, y: float, size: int, width: int, height: int) -> tuple[int, int] | None:
    """Clamp a top-left corner so the patch stays inside the image."""
    if size > width or size > height:
        return None
    return (
        min(max(int(round(x)), 0), width - size),
        min(max(int(round(y)), 0), height - size),
    )


# -- sampling ----------------------------------------------------------------

@dataclass(frozen=True)
class SamplerConfig:
    patch_size: int
    patches_per_image: int
    margin: float
    weights: tuple[tuple[str, float], ...]
    edge_jitter: float
    min_zone_frac: float
    min_center_distance: float
    max_tries: int
    backfill: bool
    max_exclusion_overlap: float
    min_std: float
    clip_to_margin: bool
    image_format: str
    quality: int
    seed: int
    dry_run: bool


def allocate(total: int, weights: tuple[tuple[str, float], ...]) -> dict[str, int]:
    """Largest-remainder split of `total` across the weighted zones."""
    names = [name for name, _ in weights]
    values = [max(0.0, value) for _, value in weights]
    if sum(values) <= 0:
        return {name: 0 for name in names}
    exact = [total * value / sum(values) for value in values]
    counts = [int(math.floor(value)) for value in exact]
    order = sorted(range(len(exact)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in order[: total - sum(counts)]:
        counts[i] += 1
    return dict(zip(names, counts))


def propose(zone: str, rng: random.Random, bbox: Box, region: Box, cfg: SamplerConfig,
            width: int, height: int) -> tuple[int, int] | None:
    size = cfg.patch_size
    if zone == "interior":
        if bbox[2] - bbox[0] < size or bbox[3] - bbox[1] < size:
            return None
        return (rng.randint(bbox[0], bbox[2] - size), rng.randint(bbox[1], bbox[3] - size))

    if zone == "edge":
        px, py, nx, ny = perimeter_point(rng, bbox)
        offset = rng.uniform(-cfg.edge_jitter, cfg.edge_jitter) * size
        return place(px + nx * offset - size / 2.0, py + ny * offset - size / 2.0, size, width, height)

    # Exterior: push a perimeter point outward by half a patch so the tile clears the
    # document by construction, then slide further out but not past the margin region.
    # Uniform rejection sampling inside R wastes most draws once B fills much of R.
    bands = (bbox[0] - region[0], region[2] - bbox[2], bbox[1] - region[1], region[3] - bbox[3])
    if max(bands) < size:
        return None  # no side of the margin band is a full patch wide
    px, py, nx, ny = perimeter_point(rng, bbox)
    if nx:
        headroom = bands[0] if nx < 0 else bands[1]
    else:
        headroom = bands[2] if ny < 0 else bands[3]
    # offset in [size/2, headroom - size/2] keeps the patch clear of B and inside R.
    offset = size / 2.0 + rng.uniform(0.0, max(0.0, headroom - size))
    # Slide along the boundary so corners are reachable too.
    tx, ty = (-ny, nx)
    slide = rng.uniform(-0.5, 0.5) * size
    return place(
        px + nx * offset + tx * slide - size / 2.0,
        py + ny * offset + ty * slide - size / 2.0,
        size,
        width,
        height,
    )


def zone_is_valid(zone: str, patch: Box, bbox: Box, region: Box, size: int,
                  min_zone_frac: float, clip_to_margin: bool) -> tuple[bool, float]:
    area = float(size * size)
    inside_frac = intersect_area(patch, bbox) / area
    in_region = intersect_area(patch, region) >= size * size
    if zone == "interior":
        return (inside_frac >= 0.999, inside_frac)
    if zone == "edge":
        straddles = min_zone_frac <= inside_frac <= 1.0 - min_zone_frac
        return (straddles and (in_region or not clip_to_margin), inside_frac)
    # Exterior is defined by the margin band, so containment in R is not optional.
    return (inside_frac == 0.0 and in_region, inside_frac)


def far_enough(patch: Box, accepted: list[Box], min_distance: float) -> bool:
    if min_distance <= 0.0:
        return True
    cx = (patch[0] + patch[2]) / 2.0
    cy = (patch[1] + patch[3]) / 2.0
    for other in accepted:
        ox = (other[0] + other[2]) / 2.0
        oy = (other[1] + other[3]) / 2.0
        if math.hypot(cx - ox, cy - oy) < min_distance:
            return False
    return True


# -- per-image worker --------------------------------------------------------

def process_row(row: dict, cfg: SamplerConfig, out_dir: str | None, row_index: int,
                overlay_dir: str | None) -> tuple[list[dict], Counter, str | None]:
    stats: Counter = Counter()
    source_path = row["source_path"]
    label = row["label"]

    def skipped(reason: str, error: str | None = None) -> tuple[list[dict], Counter, str | None]:
        # Counted per label as well, so a skip that hits one class only is visible
        # in the summary instead of silently producing a one-class dataset.
        return ([], Counter({reason: 1, f"{reason}__label_{label}": 1}), error)

    try:
        image = Image.open(source_path)
        width, height = image.size
    except Exception as error:  # unreadable / truncated / missing
        return skipped("rows_failed", f"{source_path}: {error}")

    bbox = row["bbox"]
    if bbox is None:
        return skipped("rows_no_bbox")
    clipped = clip_box(bbox, width, height)
    if clipped is None:
        return skipped("rows_bbox_outside_image")
    region = expand_bbox(bbox, cfg.margin, width, height)
    if region is None:
        return skipped("rows_bbox_outside_image")
    if cfg.patch_size > width or cfg.patch_size > height:
        return skipped("rows_image_too_small")

    # blake2b, not hash(): str hashing is salted per interpreter, which would make
    # --seed non-reproducible across runs.
    digest = hashlib.blake2b(f"{cfg.seed}|{row_index}|{source_path}".encode("utf-8"), digest_size=8)
    rng = random.Random(int.from_bytes(digest.digest(), "big"))
    exclusions = [box for box in (clip_box(b, width, height) for b in row["exclusions"]) if box]
    size = cfg.patch_size
    min_distance = cfg.min_center_distance * size

    # Decoding is deferred until a patch is actually accepted.
    pixels = None
    if cfg.min_std > 0.0 and not cfg.dry_run:
        image = image.convert("RGB")
        pixels = image

    accepted_boxes: list[Box] = []
    records: list[dict] = []

    def try_zone(zone: str) -> bool:
        nonlocal pixels
        for _ in range(cfg.max_tries):
            corner = propose(zone, rng, clipped, region, cfg, width, height)
            if corner is None:
                stats[f"reject_{zone}_no_room"] += 1
                return False
            patch = (corner[0], corner[1], corner[0] + size, corner[1] + size)

            valid, inside_frac = zone_is_valid(
                zone, patch, clipped, region, size, cfg.min_zone_frac, cfg.clip_to_margin
            )
            if not valid:
                stats[f"reject_{zone}_zone"] += 1
                continue
            if not far_enough(patch, accepted_boxes, min_distance):
                stats[f"reject_{zone}_too_close"] += 1
                continue
            if exclusions:
                worst = max(intersect_area(patch, box) for box in exclusions) / float(size * size)
                if worst > cfg.max_exclusion_overlap:
                    stats[f"reject_{zone}_excluded"] += 1
                    continue
            if cfg.min_std > 0.0 and not cfg.dry_run:
                if pixels is None:
                    pixels = image.convert("RGB")
                tile = np.asarray(pixels.crop(patch).convert("L"), dtype=np.float32)
                if float(tile.std()) < cfg.min_std:
                    stats[f"reject_{zone}_flat"] += 1
                    continue

            patch_id = f"{rng.getrandbits(128):032x}"
            accepted_boxes.append(patch)
            records.append(
                {
                    "patch_id": patch_id,
                    "relative_path": f"patches/{row['label']}/{patch_id[:2]}/{patch_id}.{cfg.image_format}",
                    "label": row["label"],
                    "zone": zone,
                    "source_path": source_path,
                    "x": patch[0],
                    "y": patch[1],
                    "size": size,
                    "inside_frac": round(inside_frac, 4),
                    "image_width": width,
                    "image_height": height,
                    "source_bbox": json.dumps([clipped[0], clipped[1], clipped[2], clipped[3]], separators=(",", ":")),
                }
            )
            stats[f"patches_{zone}"] += 1
            return True
        stats[f"reject_{zone}_exhausted"] += 1
        return False

    targets = allocate(cfg.patches_per_image, cfg.weights)
    workable = {zone for zone in ZONES if targets.get(zone, 0) > 0}
    for zone in ZONES:
        for _ in range(targets.get(zone, 0)):
            if not try_zone(zone):
                workable.discard(zone)
                break

    if cfg.backfill:
        candidates = [zone for zone in ZONES if zone in workable]
        while len(records) < cfg.patches_per_image and candidates:
            progressed = False
            for zone in list(candidates):
                if len(records) >= cfg.patches_per_image:
                    break
                if try_zone(zone):
                    stats["patches_backfilled"] += 1
                    progressed = True
                else:
                    candidates.remove(zone)
            if not progressed:
                break

    if len(records) < cfg.patches_per_image:
        stats["rows_short"] += 1
        stats["patches_missing"] += cfg.patches_per_image - len(records)
    if not records:
        stats["rows_empty"] += 1
    else:
        stats["rows_ok"] += 1

    if overlay_dir and records:
        write_overlay(image, clipped, region, records, Path(overlay_dir) / f"{row_index:06d}.jpg")

    if not cfg.dry_run and out_dir and records:
        try:
            if pixels is None:
                pixels = image.convert("RGB")
            root = Path(out_dir)
            for record in records:
                target = root / record["relative_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                tile = pixels.crop(
                    (record["x"], record["y"], record["x"] + record["size"], record["y"] + record["size"])
                )
                if cfg.image_format == "png":
                    tile.save(target, format="PNG", compress_level=6)
                else:
                    tile.save(target, format="JPEG", quality=cfg.quality, subsampling=0)
        except Exception as error:  # truncated file, decode failure part-way through
            image.close()
            return ([], Counter({"rows_failed": 1}), f"{source_path}: {error}")

    image.close()
    return (records, stats, None)


def write_overlay(image: Image.Image, bbox: Box, region: Box, records: list[dict], target: Path) -> None:
    """Annotated copy of the source image. Contains PII -- keep in the trusted zone."""
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    line = max(2, canvas.width // 400)
    draw.rectangle(region, outline=(255, 255, 255), width=line)
    draw.rectangle(bbox, outline=(60, 255, 60), width=line * 2)
    for record in records:
        box = (record["x"], record["y"], record["x"] + record["size"], record["y"] + record["size"])
        draw.rectangle(box, outline=ZONE_COLOURS[record["zone"]], width=line)
    if canvas.width > 1600:
        scale = 1600 / canvas.width
        canvas = canvas.resize((1600, int(canvas.height * scale)), Image.BILINEAR)
    canvas.save(target, format="JPEG", quality=85)


# -- dataset sanity ----------------------------------------------------------

def skips_by_label(stats: Counter) -> dict[str, dict[str, int]]:
    """{label: {reason: count}} from the per-label skip counters."""
    table: dict[str, dict[str, int]] = {}
    for key, count in stats.items():
        if "__label_" not in key:
            continue
        reason, _, label = key.partition("__label_")
        table.setdefault(label, {})[reason] = count
    return {label: dict(sorted(reasons.items())) for label, reasons in sorted(table.items())}


def check_class_balance(rows: list[dict], records: list[dict], stats: Counter) -> list[str]:
    """Catch the failure modes that produce a dataset which trains to a constant.

    A one-class output is FATAL. A skip reason that removes a much larger share
    of one class than the other is a warning: the model would learn the skip.
    """
    problems: list[str] = []
    rows_in = Counter(row["label"] for row in rows)
    patches_out = Counter(record["label"] for record in records)

    present = sorted(patches_out)
    if len(rows_in) >= 2 and len(present) < 2:
        lost = sorted(set(rows_in) - set(present))
        problems.append(
            f"FATAL: output contains only label {present} -- every row with label {lost} was "
            f"skipped ({rows_in[lost[0]]} input rows). Check skips_by_label."
        )
    elif not present:
        problems.append("FATAL: no patches produced at all.")

    for reason in sorted({k.partition("__label_")[0] for k in stats if "__label_" in k}):
        rates = {}
        for label, total in rows_in.items():
            if total:
                rates[label] = stats.get(f"{reason}__label_{label}", 0) / total
        if len(rates) >= 2 and max(rates.values()) - min(rates.values()) > 0.20:
            detail = ", ".join(f"label {l}: {r:.0%}" for l, r in sorted(rates.items()))
            problems.append(
                f"WARNING: {reason} is label-correlated ({detail}). A classifier will learn "
                f"whatever differs between the surviving rows."
            )

    if len(present) == 2:
        share = patches_out[present[0]] / max(1, sum(patches_out.values()))
        if share < 0.2 or share > 0.8:
            problems.append(
                f"WARNING: output is {share:.0%} label {present[0]} -- strongly imbalanced."
            )
    return problems


# -- CLI ---------------------------------------------------------------------

def parse_weights(text: str) -> tuple[tuple[str, float], ...]:
    weights = {zone: 0.0 for zone in ZONES}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, value = item.partition("=")
        name = name.strip()
        if name not in weights:
            raise argparse.ArgumentTypeError(f"Unknown zone {name!r}, expected one of {ZONES}")
        weights[name] = float(value)
    if sum(weights.values()) <= 0:
        raise argparse.ArgumentTypeError("At least one zone weight must be positive")
    return tuple((zone, weights[zone]) for zone in ZONES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample anonymising texture patches from bbox-cropped document images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_csv", type=Path, nargs="+", help="path,label,bbox CSV(s)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help="Patch side in SOURCE pixels. Matches IMAGE_SIZE in train_efficientnet_b2.py "
        "so the training resize is a no-op and moire survives untouched.",
    )
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument(
        "--margin",
        type=float,
        default=25.0,
        help="Percent of bbox width/height added on each side, as in crop_by_bbox(). "
        "Bounds `exterior` patches only; `edge` patches are bounded by the image.",
    )
    parser.add_argument("--zone-weights", type=parse_weights, default="interior=0.5,edge=0.5,exterior=0.0")
    parser.add_argument(
        "--edge-jitter",
        type=float,
        default=0.2,
        help="Edge patch centres are offset along the boundary normal by up to this "
        "fraction of the patch size, so the boundary is not always dead centre.",
    )
    parser.add_argument(
        "--min-zone-frac",
        type=float,
        default=0.15,
        help="An edge patch must have at least this fraction of its area on each side "
        "of the document boundary.",
    )
    parser.add_argument("--min-center-distance", type=float, default=0.25,
                        help="Minimum distance between patch centres, as a fraction of patch size.")
    parser.add_argument("--max-tries", type=int, default=40, help="Placement attempts per patch.")
    parser.add_argument("--no-backfill", action="store_true",
                        help="Do not top up from other zones when a zone cannot reach its quota.")
    parser.add_argument("--exclude-column", type=str, default=None,
                        help="CSV column holding a JSON list of [x1,y1,x2,y2] PII regions "
                             "(portrait, MRZ, signature) that patches must avoid.")
    parser.add_argument("--max-exclusion-overlap", type=float, default=0.0,
                        help="Allowed fraction of a patch overlapping an exclusion region.")
    parser.add_argument(
        "--min-std",
        type=float,
        default=0.0,
        help="Reject patches whose greyscale std is below this. OFF by default: flat "
        "regions still carry sensor noise and glare, and the rejection rate can differ "
        "between bona-fide and attack images, which would make the filter label-correlated.",
    )
    parser.add_argument(
        "--clip-to-margin",
        action="store_true",
        help="Also confine `edge` patches to the margin region. Off by default, so edge "
        "patches keep working when the margin band is narrower than half a patch; "
        "with a small --margin they will then extend past it, bounded by the image. "
        "`exterior` patches are always confined to the margin region.",
    )
    parser.add_argument("--format", dest="image_format", choices=("png", "jpeg"), default="png",
                        help="PNG is lossless. JPEG re-encoding overwrites the compression fingerprint.")
    parser.add_argument("--quality", type=int, default=100, help="JPEG quality when --format jpeg.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N rows (0 = all).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overlay-dir", type=Path, default=None,
                        help="Write annotated source images for visual QA. These contain PII.")
    parser.add_argument("--overlay-limit", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true", help="Plan and report without writing patches.")
    parser.add_argument("--allow-single-class", action="store_true",
                        help="Exit 0 even when the output contains only one label.")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.zone_weights if isinstance(args.zone_weights, tuple) else parse_weights(args.zone_weights)

    rows: list[dict] = []
    for csv_path in args.input_csv:
        rows.extend(read_input_csv(csv_path, args.exclude_column))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No rows to process")

    cfg = SamplerConfig(
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        margin=args.margin,
        weights=weights,
        edge_jitter=args.edge_jitter,
        min_zone_frac=args.min_zone_frac,
        min_center_distance=args.min_center_distance,
        max_tries=args.max_tries,
        backfill=not args.no_backfill,
        max_exclusion_overlap=args.max_exclusion_overlap,
        min_std=args.min_std,
        clip_to_margin=args.clip_to_margin,
        image_format=args.image_format,
        quality=args.quality,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = str(args.overlay_dir) if args.overlay_dir else None

    stats: Counter = Counter()
    records: list[dict] = []
    errors: list[str] = []

    print(f"Rows: {len(rows)}  patch={args.patch_size}px  per-image={args.patches_per_image}  "
          f"zones={dict(weights)}  margin={args.margin}%")

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                process_row,
                row,
                cfg,
                None if args.dry_run else str(out_dir),
                index,
                overlay_dir if index < args.overlay_limit else None,
            ): index
            for index, row in enumerate(rows)
        }
        done = 0
        for future in as_completed(futures):
            row_records, row_stats, error = future.result()
            records.extend(row_records)
            stats.update(row_stats)
            if error:
                errors.append(error)
                if args.fail_fast:
                    raise SystemExit(error)
            done += 1
            if done % 500 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)} rows, {len(records)} patches", file=sys.stderr)

    # Shuffle so neither file order nor CSV order groups patches by source document.
    random.Random(args.seed).shuffle(records)

    problems = check_class_balance(rows, records, stats)

    patches_csv = out_dir / "patches.csv"
    manifest_csv = out_dir / "manifest.csv"
    summary_json = out_dir / "summary.json"

    if not args.dry_run:
        with patches_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["path", "label", "zone"])
            for record in records:
                writer.writerow([record["relative_path"], record["label"], record["zone"]])

        manifest_fields = ["patch_id", "relative_path", "label", "zone", "source_path",
                           "x", "y", "size", "inside_frac", "image_width", "image_height", "source_bbox"]
        with manifest_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=manifest_fields)
            writer.writeheader()
            writer.writerows(records)

    summary = {
        "rows_total": len(rows),
        "rows_by_label": {str(k): v for k, v in sorted(Counter(row["label"] for row in rows).items())},
        "patches_total": len(records),
        "by_zone": dict(Counter(record["zone"] for record in records)),
        "by_label": {str(k): v for k, v in sorted(Counter(record["label"] for record in records).items())},
        "skips_by_label": skips_by_label(stats),
        "problems": problems,
        "counters": dict(sorted(stats.items())),
        "errors": len(errors),
        "config": {
            "patch_size": cfg.patch_size,
            "patches_per_image": cfg.patches_per_image,
            "margin": cfg.margin,
            "zone_weights": dict(weights),
            "edge_jitter": cfg.edge_jitter,
            "min_zone_frac": cfg.min_zone_frac,
            "min_center_distance": cfg.min_center_distance,
            "max_exclusion_overlap": cfg.max_exclusion_overlap,
            "min_std": cfg.min_std,
            "clip_to_margin": cfg.clip_to_margin,
            "format": cfg.image_format,
            "seed": cfg.seed,
            "inputs": [str(p) for p in args.input_csv],
        },
    }
    if not args.dry_run:
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if errors:
        print(f"\n{len(errors)} unreadable source images, first 10:", file=sys.stderr)
        for error in errors[:10]:
            print(f"  {error}", file=sys.stderr)

    if args.dry_run:
        print("\nDry run: no patches written.")
    else:
        print(f"\nSafe to move : {patches_csv}")
        print(f"KEEP BEHIND  : {manifest_csv}  (re-identification map)")
        if overlay_dir:
            print(f"KEEP BEHIND  : {overlay_dir}  (annotated originals)")

    if problems:
        print("\nPROBLEMS:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        fatal = any(p.startswith("FATAL") for p in problems)
        if fatal and not args.allow_single_class:
            print("\nRefusing to hand over a dataset a classifier cannot learn from. "
                  "Fix the bboxes (run the detector on the missing rows) or pass "
                  "--allow-single-class if this is intentional.", file=sys.stderr)
            raise SystemExit(2)


if __name__ == "__main__":
    main()
