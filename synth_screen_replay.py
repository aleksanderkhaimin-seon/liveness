#!/usr/bin/env python3
"""Turn synthetic "document + background" frames into screen-replay look-alikes.

The goal is the coarse signal that survives downscaling to the ~96 px document
export, not pixel-level realism: display colour response, backlight black level,
glare, bezel, light spill onto the surroundings, banding, bloom, and then the
camera that re-captured the screen (defocus, noise, vignette, JPEG). Moire and a
subpixel grid are included but are weak by default because they vanish at
export scale.

Screen extent (--mode):

    doc     the document rectangle (bbox) is the screen content; a bezel is drawn
            around it and the surroundings get dimmed / light spill. Needs bbox.
    frame   the whole frame is the screen content (a full-screen replay).
    mixed   per image, doc with probability --doc-prob, else frame.

Class-leak control. The generator's fingerprint must not tell the classes apart.
Run the live synthetic frames through --camera-only (same camera, noise, JPEG,
vignette and tilt, no screen effects) so the only difference between classes is
the screen. Output label comes from --label (default 0 = attack); use --label 1
with --camera-only for the live control.

Input is a CSV with path[,label][,bbox] (bbox = JSON [x1,y1,x2,y2]) or a folder
of images. Output is images plus a CSV path,label,bbox[,params].

Examples:

    python synth_screen_replay.py data/synth_docs.csv --out-dir /data/synth/replay --mode mixed --workers 16
    python synth_screen_replay.py data/synth_docs.csv --out-dir /data/synth/live --camera-only --label 1
    python synth_screen_replay.py data/synth_docs.csv --out-dir /tmp/preview --limit 12 --preview-sheet
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# -- helpers ------------------------------------------------------------------

def rng_for(seed: int, key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def parse_bbox(raw: str, width: int, height: int) -> tuple[int, int, int, int] | None:
    raw = (raw or "").strip()
    if raw.lower() in {"", "none", "null", "nan"}:
        return None
    x1, y1, x2, y2 = (float(v) for v in json.loads(raw))
    x1, x2 = sorted((min(max(x1, 0), width), min(max(x2, 0), width)))
    y1, y2 = sorted((min(max(y1, 0), height), min(max(y2, 0), height)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return int(x1), int(y1), int(x2), int(y2)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.05:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def soft_mask(shape: tuple[int, int], box: tuple[int, int, int, int], feather: float) -> np.ndarray:
    mask = np.zeros(shape, np.float32)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 1.0
    return blur(mask, feather) if feather > 0.05 else mask


# -- screen effects (operate on float32 RGB in [0, 1], in place of the screen area) -----

def display_response(img: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Colour response of the panel: gamma, contrast, backlight black level, white point, saturation."""
    lin = srgb_to_linear(img)
    gamma = float(np.exp(rng.normal(0.0, 0.12 * strength)))
    lin = lin ** gamma
    brightness = float(rng.uniform(0.45, 1.0)) ** strength
    black_level = float(rng.uniform(0.0, 0.06)) * strength  # LCD backlight bleed lifts blacks
    lin = black_level + (1.0 - black_level) * lin * brightness
    # white point: warm (night mode) to cool (blue-heavy panel)
    temp = float(rng.normal(0.0, 0.08 * strength))
    lin = lin * np.array([1.0 + temp, 1.0, 1.0 - temp], np.float32)
    out = linear_to_srgb(lin)
    gray = out.mean(axis=2, keepdims=True)
    sat = float(rng.uniform(0.75, 1.15))
    out = gray + (out - gray) * sat
    return np.clip(out, 0, 1)


def pixel_grid(shape: tuple[int, int], rng: np.random.Generator, scale: float) -> np.ndarray:
    """RGB stripe subpixel texture plus a low-frequency moire beat, multiplicative around 1.0."""
    h, w = shape
    period = float(rng.uniform(2.0, 4.5)) * scale
    angle = float(rng.uniform(-8, 8)) * math.pi / 180
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx * math.cos(angle) + yy * math.sin(angle)
    v = -xx * math.sin(angle) + yy * math.cos(angle)
    grid = np.zeros((h, w, 3), np.float32)
    for c in range(3):  # R, G, B stripes shifted by a third of the period
        grid[..., c] = 0.5 + 0.5 * np.cos(2 * math.pi * (u / period - c / 3.0))
    amp_grid = float(rng.uniform(0.0, 0.08))
    out = 1.0 - amp_grid + amp_grid * 2 * grid
    beat_f = float(rng.uniform(1 / 60, 1 / 14))
    beat_angle = float(rng.uniform(0, math.pi))
    beat = np.cos(2 * math.pi * beat_f * (xx * math.cos(beat_angle) + yy * math.sin(beat_angle))) \
        * np.cos(2 * math.pi * beat_f * 0.8 * (v + 0.4 * u))
    amp_moire = float(rng.uniform(0.0, 0.06))
    return out * (1.0 + amp_moire * beat[..., None])


def banding(shape: tuple[int, int], rng: np.random.Generator, scale: float) -> np.ndarray:
    """Exposure / refresh beat: broad, slowly drifting horizontal brightness bands."""
    h, w = shape
    freq = float(rng.uniform(1.5, 7.0)) / h
    phase = float(rng.uniform(0, 2 * math.pi))
    tilt = float(rng.uniform(-0.15, 0.15))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    band = np.cos(2 * math.pi * freq * (yy + tilt * xx) + phase)
    amp = float(rng.uniform(0.0, 0.10))
    return (1.0 + amp * band)[..., None]


def glare_layer(shape: tuple[int, int], box: tuple[int, int, int, int], rng: np.random.Generator) -> np.ndarray:
    """Additive reflection over the screen: a soft blob, a diagonal sheen, or a window-shaped patch."""
    h, w = shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    layer = np.zeros((h, w), np.float32)
    kind = rng.choice(["blob", "sheen", "window", "none"], p=[0.35, 0.25, 0.15, 0.25])
    if kind == "blob":
        cx = rng.uniform(x1, x2)
        cy = rng.uniform(y1, y2)
        sx = rng.uniform(0.12, 0.45) * bw
        sy = rng.uniform(0.12, 0.45) * bh
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        layer = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        layer *= rng.uniform(0.15, 0.7)
    elif kind == "sheen":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        angle = rng.uniform(0, math.pi)
        d = (xx - x1) * math.cos(angle) + (yy - y1) * math.sin(angle)
        centre = rng.uniform(0.1, 0.9) * (abs(bw * math.cos(angle)) + abs(bh * math.sin(angle)))
        width = rng.uniform(0.08, 0.35) * max(bw, bh)
        layer = np.exp(-((d - centre) / width) ** 2) * rng.uniform(0.1, 0.5)
    elif kind == "window":
        rx1 = rng.uniform(x1, x1 + 0.6 * bw)
        ry1 = rng.uniform(y1, y1 + 0.6 * bh)
        rx2 = min(x2, rx1 + rng.uniform(0.2, 0.6) * bw)
        ry2 = min(y2, ry1 + rng.uniform(0.2, 0.6) * bh)
        layer = soft_mask((h, w), (int(rx1), int(ry1), int(rx2), int(ry2)), 0.04 * max(bw, bh))
        layer *= rng.uniform(0.08, 0.3)
    return layer


def bloom(img: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    bright = np.clip(img - 0.7, 0, 1)
    glow = blur(bright, float(rng.uniform(4, 14)) * scale)
    return img + glow * float(rng.uniform(0.0, 0.6))


def add_bezel(img: np.ndarray, box: tuple[int, int, int, int], rng: np.random.Generator) -> np.ndarray:
    """Replace the surroundings with a bezel strip, shaded like a device edge."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    long_side = max(x2 - x1, y2 - y1)
    thick = int(rng.uniform(0.015, 0.07) * long_side)
    if thick < 2:
        return img
    colour = rng.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
    base = {0: np.array([0.03, 0.03, 0.035]), 1: np.array([0.55, 0.56, 0.58]), 2: np.array([0.85, 0.85, 0.83])}[colour]
    base = (base * rng.uniform(0.7, 1.3)).astype(np.float32)
    bx1, by1 = max(0, x1 - thick), max(0, y1 - thick)
    bx2, by2 = min(w, x2 + thick), min(h, y2 + thick)
    bezel = np.zeros((h, w, 3), np.float32) + base
    # brushed gradient along the long edge so it is not a flat fill
    grad = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(1.0, 1.15), w, dtype=np.float32)
    bezel *= grad[None, :, None]
    mask = soft_mask((h, w), (bx1, by1, bx2, by2), max(0.6, thick * 0.12))[..., None]
    inner = soft_mask((h, w), box, 0.8)[..., None]
    ring = np.clip(mask - inner, 0, 1)  # strip only; the glass keeps the screen content
    img = img * (1 - ring) + bezel * ring
    shadow = blur(inner[..., 0], thick * 0.6 + 1)[..., None]  # inner edge shadow where the glass meets the frame
    return img * (1 - 0.35 * np.clip(shadow - inner, 0, 1))


def light_spill(img: np.ndarray, screen_mask: np.ndarray, rng: np.random.Generator, scale_px: float) -> np.ndarray:
    """Screen light on the surroundings (cool cast), plus the room being dimmer than the screen."""
    room = float(rng.uniform(0.55, 1.0))
    outside = (1 - screen_mask)[..., None]
    img = img * (1 - outside * (1 - room))
    mean = (img * screen_mask[..., None]).sum((0, 1)) / max(screen_mask.sum(), 1.0)
    spill = blur(screen_mask, float(rng.uniform(0.15, 0.5)) * scale_px)[..., None] * mean
    return img + spill * outside * float(rng.uniform(0.0, 0.25))


# -- camera (applied to the whole frame; also used alone for the live control) ----------

def vignette(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    return img * (1 - float(rng.uniform(0.0, 0.35)) * np.clip(r2 / 2, 0, 1))[..., None]


def sensor_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shot = float(rng.uniform(0.0, 0.025))
    read = float(rng.uniform(0.0, 0.015))
    chroma = float(rng.uniform(0.0, 0.01))
    noise = rng.normal(0, 1, img.shape).astype(np.float32)
    out = img + noise * np.sqrt(np.clip(img, 0, 1) * shot ** 2 + read ** 2)
    out += rng.normal(0, 1, img.shape).astype(np.float32) * chroma
    return out


def jpeg_roundtrip(img_u8: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img_u8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1] if ok else img_u8


def tilt(img: np.ndarray, box, rng: np.random.Generator, max_frac: float):
    """Small global perspective jitter; returns image and the transformed axis-aligned bbox."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-max_frac, max_frac, (4, 2)) * np.array([w, h])
    dst = (src + jitter).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if box is None:
        return out, None
    x1, y1, x2, y2 = box
    corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])[None]
    moved = cv2.perspectiveTransform(corners, matrix)[0]
    nx1, ny1 = np.clip(moved.min(0), 0, [w, h])
    nx2, ny2 = np.clip(moved.max(0), 0, [w, h])
    return out, (int(nx1), int(ny1), int(nx2), int(ny2))


def camera(img: np.ndarray, box, rng: np.random.Generator, args) -> tuple[np.ndarray, tuple | None, dict]:
    h, w = img.shape[:2]
    scale = max(h, w) / 1000.0
    params: dict = {}
    if args.tilt > 0:
        img, box = tilt(img, box, rng, args.tilt)
    sigma = float(rng.uniform(0.0, args.blur)) * scale
    img = blur(img, sigma)
    params["blur"] = round(sigma, 2)
    # recapture resampling: camera sensor is lower-res than the display
    if rng.random() < 0.5:
        f = float(rng.uniform(0.5, 0.9))
        small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        params["resample"] = round(f, 2)
    img = vignette(img, rng)
    img = sensor_noise(img, rng)
    img_u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    quality = int(rng.integers(args.jpeg_min, args.jpeg_max + 1))
    params["jpeg"] = quality
    return jpeg_roundtrip(img_u8, quality), box, params


# -- full pipeline ---------------------------------------------------------------------

def replay(img_u8: np.ndarray, box, rng: np.random.Generator, args) -> tuple[np.ndarray, tuple | None, dict]:
    h, w = img_u8.shape[:2]
    img = img_u8.astype(np.float32) / 255.0
    scale = max(h, w) / 1000.0
    mode = args.mode
    if mode == "mixed":
        mode = "doc" if (box is not None and rng.random() < args.doc_prob) else "frame"
    if mode == "doc" and box is None:
        raise ValueError("--mode doc needs a bbox for every row")
    screen_box = box if mode == "doc" else (0, 0, w, h)
    params = {"mode": mode}

    screen_mask = soft_mask((h, w), screen_box, 0.6 * scale)
    m3 = screen_mask[..., None]

    # content blurs a little: the displayed image is itself resampled by the panel
    shown = blur(img, float(rng.uniform(0.0, 0.8)) * scale)
    shown = display_response(shown, rng, args.strength)
    shown = shown * pixel_grid((h, w), rng, scale * args.grid_scale) * banding((h, w), rng, scale)
    shown = shown + glare_layer((h, w), screen_box, rng)[..., None] * np.array([1.0, 1.0, 1.0], np.float32)
    shown = bloom(np.clip(shown, 0, 1), rng, scale)

    if mode == "doc":
        img = img * (1 - m3) + shown * m3
        img = light_spill(img, screen_mask, rng, max(box[2] - box[0], box[3] - box[1]) * 0.2)
        if rng.random() < args.bezel_prob:
            img = add_bezel(img, box, rng)
            params["bezel"] = True
    else:
        img = shown
    return camera_then_pack(np.clip(img, 0, 1), box, rng, args, params)


def camera_then_pack(img, box, rng, args, params):
    img_u8, box, cam = camera(img, box, rng, args)
    params.update(cam)
    return img_u8, box, params


def process(job: tuple) -> dict:
    index, src, label, bbox_raw, out_dir, args_dict = job
    args = argparse.Namespace(**args_dict)
    row = {"src": src, "ok": False}
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        row["error"] = "unreadable"
        return row
    img = img[..., ::-1]
    h, w = img.shape[:2]
    try:
        box = parse_bbox(bbox_raw, w, h)
    except (ValueError, json.JSONDecodeError):
        box = None
    rng = rng_for(args.seed, f"{index}:{src}")
    try:
        if args.camera_only:
            out, box, params = camera(img.astype(np.float32) / 255.0, box, rng, args)
        else:
            out, box, params = replay(img, box, rng, args)
    except ValueError as error:
        row["error"] = str(error)
        return row
    name = f"{index:07d}_{hashlib.sha1(src.encode()).hexdigest()[:8]}.jpg"
    path = Path(out_dir) / name
    cv2.imwrite(str(path), out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    row.update(ok=True, path=str(path), label=label, bbox=json.dumps(list(box)) if box else "", params=json.dumps(params))
    return row


# -- IO ---------------------------------------------------------------------------------

def read_inputs(source: Path) -> list[tuple[str, str]]:
    if source.is_dir():
        return [(str(p), "") for p in sorted(source.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
    rows = []
    with source.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise SystemExit(f"{source} needs a header with a path column")
        for r in reader:
            path = Path(r["path"].strip())
            if not path.is_absolute():
                path = source.parent / path
            rows.append((str(path), r.get("bbox", "") or ""))
    return rows


def preview_sheet(rows: list[dict], out_path: Path, originals: dict[str, str]) -> None:
    tiles = []
    for r in rows:
        a = cv2.imread(originals[r["src"]])
        b = cv2.imread(r["path"])
        if a is None or b is None:
            continue
        size = 320
        a = cv2.resize(a, (size, int(size * a.shape[0] / a.shape[1])))
        b = cv2.resize(b, (size, int(size * b.shape[0] / b.shape[1])))
        tile = np.zeros((max(a.shape[0], b.shape[0]), size * 2 + 6, 3), np.uint8)
        tile[: a.shape[0], :size] = a
        tile[: b.shape[0], size + 6:] = b
        tiles.append(tile)
    if not tiles:
        return
    cols = 2
    cell_h = max(t.shape[0] for t in tiles)
    cell_w = tiles[0].shape[1]
    grid_rows = math.ceil(len(tiles) / cols)
    sheet = np.full((grid_rows * (cell_h + 6), cols * (cell_w + 12), 3), 40, np.uint8)
    for i, t in enumerate(tiles):
        y, x = (i // cols) * (cell_h + 6), (i % cols) * (cell_w + 12)
        sheet[y: y + t.shape[0], x: x + t.shape[1]] = t
    cv2.imwrite(str(out_path), sheet)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="CSV (path[,label][,bbox]) or folder of images.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--label", type=int, choices=(0, 1), default=0, help="Label written to the output CSV (default 0 = attack).")
    ap.add_argument("--mode", choices=("doc", "frame", "mixed"), default="mixed")
    ap.add_argument("--doc-prob", type=float, default=0.6, help="mixed mode: probability of doc-as-screen.")
    ap.add_argument("--camera-only", action="store_true", help="Skip screen effects; camera only (live control).")
    ap.add_argument("--strength", type=float, default=1.0, help="Scales the colour-response variation.")
    ap.add_argument("--bezel-prob", type=float, default=0.7)
    ap.add_argument("--grid-scale", type=float, default=1.0, help="Subpixel grid period multiplier.")
    ap.add_argument("--tilt", type=float, default=0.03, help="Corner jitter as a fraction of frame size (0 disables).")
    ap.add_argument("--blur", type=float, default=1.4, help="Max camera defocus sigma at 1000 px frame size.")
    ap.add_argument("--jpeg-min", type=int, default=55)
    ap.add_argument("--jpeg-max", type=int, default=92)
    ap.add_argument("--copies", type=int, default=1, help="Variants per input frame, each with its own randomness.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-params", action="store_true", help="Add a params column (debugging; do not train on it).")
    ap.add_argument("--preview-sheet", action="store_true", help="Write preview.jpg with original | result pairs.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    inputs = read_inputs(args.source)
    if args.limit:
        inputs = inputs[: args.limit]
    if not inputs:
        raise SystemExit("no input rows")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args_dict = {k: v for k, v in vars(args).items() if k != "source"}
    jobs = []
    for copy in range(args.copies):
        for i, (src, bbox) in enumerate(inputs):
            jobs.append((copy * len(inputs) + i, src, args.label, bbox, str(args.out_dir), args_dict))

    results = []
    progress = tqdm(total=len(jobs), desc="replay", unit="img")
    try:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for r in pool.map(process, jobs, chunksize=16):
                    results.append(r)
                    progress.update(1)
        else:
            for job in jobs:
                results.append(process(job))
                progress.update(1)
    finally:
        progress.close()

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    fields = ["path", "label", "bbox"] + (["params"] if args.save_params else [])
    with (args.out_dir / "frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ok)
    # source mapping stays behind with the generator; the training CSV does not need it
    with (args.out_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "source"])
        writer.writerows((r["path"], r["src"]) for r in ok)
    if args.preview_sheet:
        preview_sheet(ok[:12], args.out_dir / "preview.jpg", {r["src"]: r["src"] for r in ok})
    print(f"wrote {len(ok)} images to {args.out_dir} ({len(failed)} skipped)")
    for r in failed[:10]:
        print(f"  skipped {r['src']}: {r.get('error')}")


if __name__ == "__main__":
    main()
