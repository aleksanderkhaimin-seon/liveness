#!/usr/bin/env python3
"""
check_data_leakage.py

Checks whether any images/frames from a TEST session also appear in one
or more TRAIN sessions. Two checks are run:

1. Exact-duplicate check (always on)
   Hashes the *decoded pixel content* of every image (not the raw file
   bytes). This catches duplicates even if the same frame was re-saved
   with different compression, filename, or metadata -- which is common
   when frames get re-exported across scraping sessions.

2. Near-duplicate check (optional, --phash)
   Uses a perceptual hash (pHash) to flag frames that are visually
   almost identical (resized, re-cropped, slightly re-compressed) but
   not byte-for-byte or pixel-for-pixel identical. This is a brute-force
   comparison (test x train), so it can be slow on very large datasets --
   see the note at the bottom of this file if you need to scale it up.

Usage:
    python check_data_leakage.py \
        --train /path/session1 /path/session2 /path/session3 /path/session4 \
        --test  /path/session5 \
        --out   leakage_report.csv \
        [--recursive] [--phash] [--phash-threshold 5]

Requires: Pillow (PIL). For --phash, also requires the `imagehash` package
    pip install pillow imagehash --break-system-packages
"""

import argparse
import hashlib
import os
import sys
import csv
from collections import defaultdict

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(folder, recursive):
    walker = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
    for root, _, files in walker:
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                yield os.path.join(root, f)


def content_hash(path):
    """Hash decoded pixel data so re-compressed/re-saved copies still match."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return hashlib.sha256(im.tobytes()).hexdigest()
    except Exception as e:
        print(f"  [warn] could not read {path}: {e}", file=sys.stderr)
        return None


def perceptual_hash(path):
    from PIL import Image
    import imagehash
    try:
        with Image.open(path) as im:
            return imagehash.phash(im)
    except Exception as e:
        print(f"  [warn] could not phash {path}: {e}", file=sys.stderr)
        return None


def build_index(session_paths, recursive, label):
    """session_paths: dict of session_name -> folder. Returns hash -> [(session, filepath), ...]"""
    index = defaultdict(list)
    for session_name, folder in session_paths.items():
        files = list(find_images(folder, recursive))
        print(f"[{label}] session '{session_name}': {len(files)} images found in {folder}")
        for path in files:
            h = content_hash(path)
            if h:
                index[h].append((session_name, path))
    return index


def main():
    ap = argparse.ArgumentParser(description="Detect train/test leakage across image sessions.")
    ap.add_argument("--train", nargs="+", required=True, help="Paths to train session folders (4 sessions)")
    ap.add_argument("--test", nargs="+", required=True, help="Path(s) to the test session folder")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--out", default="leakage_report.csv", help="Output CSV path")
    ap.add_argument("--phash", action="store_true", help="Also run a near-duplicate perceptual-hash check")
    ap.add_argument("--phash-threshold", type=int, default=5, help="Max Hamming distance to flag as near-duplicate")
    args = ap.parse_args()

    train_paths = {f"train_{i+1}_{os.path.basename(p.rstrip('/'))}": p for i, p in enumerate(args.train)}
    test_paths = {f"test_{i+1}_{os.path.basename(p.rstrip('/'))}": p for i, p in enumerate(args.test)}

    print("== Indexing train sessions (exact/content hash) ==")
    train_index = build_index(train_paths, args.recursive, "train")

    print("\n== Scanning test session(s) for exact matches ==")
    exact_leaks = []
    test_files_by_session = {}
    for session_name, folder in test_paths.items():
        files = list(find_images(folder, args.recursive))
        test_files_by_session[session_name] = files
        print(f"[test] session '{session_name}': {len(files)} images found in {folder}")
        for path in files:
            h = content_hash(path)
            if h and h in train_index:
                for train_session, train_path in train_index[h]:
                    exact_leaks.append({
                        "type": "exact",
                        "hash_or_distance": h,
                        "test_session": session_name,
                        "test_file": path,
                        "train_session": train_session,
                        "train_file": train_path,
                    })

    print(f"\nExact-duplicate leaks found: {len(exact_leaks)}")

    near_leaks = []
    if args.phash:
        print("\n== Running perceptual-hash near-duplicate check (this can be slow) ==")
        train_phashes = []
        for session_name, folder in train_paths.items():
            for path in find_images(folder, args.recursive):
                ph = perceptual_hash(path)
                if ph is not None:
                    train_phashes.append((session_name, path, ph))

        for session_name, files in test_files_by_session.items():
            for path in files:
                ph = perceptual_hash(path)
                if ph is None:
                    continue
                for train_session, train_path, train_ph in train_phashes:
                    dist = ph - train_ph
                    if dist <= args.phash_threshold:
                        near_leaks.append({
                            "type": "near_duplicate",
                            "hash_or_distance": dist,
                            "test_session": session_name,
                            "test_file": path,
                            "train_session": train_session,
                            "train_file": train_path,
                        })
        print(f"Near-duplicate candidates found: {len(near_leaks)}")

    all_rows = exact_leaks + near_leaks
    if all_rows:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["type", "hash_or_distance", "test_session", "test_file", "train_session", "train_file"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nReport written to {args.out} ({len(all_rows)} rows)")
    else:
        print("\nNo leakage detected. No report file written.")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Notes for scaling up:
#
# - If a "session" is video, extract frames first (e.g. with ffmpeg) into
#   per-session image folders, then point this script at those folders.
#
# - Content hashing (pixel-based) already protects you against the most
#   common accidental leak: the same frame exported twice with different
#   compression/format. It will NOT catch a frame that was cropped,
#   resized, rotated, or re-shot of the same subject/moment -- for that,
#   use --phash, or better, an embedding-similarity check (e.g. cosine
#   similarity between face embeddings) if leakage risk is about the same
#   *subject* appearing in both train and test rather than the same exact
#   image content.
#
# - --phash brute-forces test x train comparisons (O(n*m)). Fine for a
#   few thousand images per session. For larger datasets, bucket by hash
#   prefix or use a library like annoy/faiss over the hash bits for
#   approximate nearest-neighbor lookup instead of the full double loop.
# ---------------------------------------------------------------------------
