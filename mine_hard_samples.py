#!/usr/bin/env python3
"""Pick hard, diverse, production-like samples from an unused pool to oversample in the next run.

Label convention: 1 = replay (attack), 0 = live; the model score is P(label 1).
Default target is label 1, i.e. replays the model takes for live.

The pool is scored once with the current model (scores and pooled embeddings in
one pass), then:

  hardness     score-error percentile within the class: 1 - score for replays,
               score for lives. Percentile, not an absolute threshold, because the
               model's EER threshold on production is nowhere near 0.5.
  band         only samples between --band-low and --band-high percentile are
               candidates. The very top slice (--band-high..1) is where mislabeled
               and unusable frames concentrate: it goes to review.csv, not to training.
  prod weight  with --prod-csv, a linear probe pool-vs-production on the embeddings;
               its odds weight replays that look like production over ones that are
               merely odd. Without it, all weights are 1.
  diversity    at most --max-per-group frames per session/video, and at most
               --max-cluster-frac of the selection from any embedding cluster.

Outputs in --out-dir:

  scored.csv    every pool row: score, hardness, percentile, group, cluster, weight
  selected.csv  path,label,bbox -- the hard sample
  control.csv   same size, random from the same pool with the same per-group cap.
                TRAIN THIS ARM TOO: without it a gain cannot be told from "more data".
  review.csv    the suspected-noise top slice, plus review.jpg with --review-sheet
  train_mix.csv with --base-csv: base rows + selected rows x --repeat
  summary.json

Never feed rows the model trained on (scores are biased) or any evaluation set:
pass them with --exclude-csv.

Example:

  python mine_hard_samples.py data/replay_pool.csv --checkpoint runs/best/best.keras \\
      --exclude-csv data/train_current.csv data/ProdTest-0.3.csv \\
      --prod-csv data/prod_replays.csv --n-select 4000 \\
      --base-csv data/train_current.csv --repeat 2 --out-dir runs/mining/r1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from progress_util import progress


# -- pure selection logic (no TensorFlow) -----------------------------------------------

def derive_group(path: str) -> str:
    """Session/video id: the folder holding the frames (one level up if it is `frames/`)."""
    parent = Path(path).parent
    return str(parent.parent if parent.name.lower() == "frames" else parent)


def hardness_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """How wrong the model is: 1 - P(replay) for replays, P(replay) for lives."""
    return np.where(labels == 1, 1.0 - scores, scores)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="stable")
    ranks = np.empty(len(values), np.float64)
    ranks[order] = np.arange(len(values)) / max(len(values) - 1, 1)
    return ranks


def production_weights(pool: np.ndarray, prod: np.ndarray, seed: int) -> np.ndarray:
    """Odds that a pool sample looks like production, cross-fitted so a sample is never scored by a probe that saw it."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.vstack([pool, prod])
    y = np.r_[np.zeros(len(pool)), np.ones(len(prod))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500, class_weight="balanced"))
    p = cross_val_predict(clf, x, y, cv=5, method="predict_proba")[:len(pool), 1]
    odds = np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))
    return np.clip(odds / np.median(odds), 0.1, 10.0)


def assign_clusters(embeddings: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    k = max(1, min(k, len(embeddings)))
    reduced = PCA(n_components=min(50, embeddings.shape[1], len(embeddings) - 1), random_state=seed).fit_transform(embeddings) \
        if len(embeddings) > 2 else embeddings
    return KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(reduced)


def weighted_diverse_pick(weights, groups, clusters, n, max_per_group, max_cluster_frac, rng) -> list[int]:
    """Weighted sampling without replacement (Gumbel top-k order) under group and cluster caps."""
    keys = np.log(np.maximum(weights, 1e-12)) + rng.gumbel(size=len(weights))
    cluster_cap = max(1, math.ceil(max_cluster_frac * n))
    per_group: dict[str, int] = {}
    per_cluster: dict[int, int] = {}
    picked: list[int] = []
    for i in np.argsort(-keys):
        g, c = groups[i], int(clusters[i])
        if per_group.get(g, 0) >= max_per_group or per_cluster.get(c, 0) >= cluster_cap:
            continue
        per_group[g] = per_group.get(g, 0) + 1
        per_cluster[c] = per_cluster.get(c, 0) + 1
        picked.append(int(i))
        if len(picked) >= n:
            break
    return picked


def random_pick_grouped(groups, n, max_per_group, rng) -> list[int]:
    per_group: dict[str, int] = {}
    picked: list[int] = []
    for i in rng.permutation(len(groups)):
        if per_group.get(groups[i], 0) >= max_per_group:
            continue
        per_group[groups[i]] = per_group.get(groups[i], 0) + 1
        picked.append(int(i))
        if len(picked) >= n:
            break
    return picked


# -- IO ---------------------------------------------------------------------------------

def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"path", "label"} <= set(reader.fieldnames):
            raise SystemExit(f"{csv_path} must have path,label columns")
        rows = []
        for r in reader:
            path = Path(r["path"].strip())
            if not path.is_absolute():
                path = csv_path.parent / path
            rows.append({"path": str(path), "orig": r["path"].strip(), "label": int(r["label"]),
                         "bbox": (r.get("bbox") or "").strip(), "group": (r.get("group") or "").strip()})
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# -- TensorFlow part: one pass for embeddings and scores ----------------------------------

def compute_embeddings(rows: list[dict], args, model, desc: str) -> tuple[np.ndarray, np.ndarray]:
    import tensorflow as tf
    from train_efficientnet_b2 import make_dataset

    head = model.get_layer("live_score")
    feature_model = tf.keras.Model(model.input, head.input)
    kernel, bias = (w.astype(np.float32) for w in head.get_weights())
    dataset = make_dataset([r["path"] for r in rows], [r["label"] for r in rows], [r["bbox"] for r in rows],
                           args.batch_size, training=False, use_bbox_crop=args.use_bbox_crop, margin=args.margin)
    total = int(math.ceil(len(rows) / args.batch_size))
    chunks = []
    for batch in progress(dataset, total, f"{desc} ({len(rows)} images)"):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        chunks.append(feature_model(images, training=False).numpy().astype(np.float32))
    feats = np.concatenate(chunks)
    logits = (feats @ kernel + bias).reshape(-1)
    return feats, 1.0 / (1.0 + np.exp(-logits))


def load_model_and_geometry(args):
    from input_geometry import find_report, resolve_geometry
    from predict_checkpoint_csv import load_model
    from train_efficientnet_b2 import configure_runtime, set_input_geometry

    configure_runtime(args.require_gpu, args.mixed_precision)
    report = find_report(args.checkpoint)
    geo = resolve_geometry(args.image_size, args.resize_mode, report_path=report)
    set_input_geometry(geo.image_size, geo.resize_mode)
    model = load_model(args.checkpoint)
    geo = resolve_geometry(args.image_size, args.resize_mode, model_input_size=int(model.input_shape[1]), report_path=report)
    set_input_geometry(geo.image_size, geo.resize_mode)
    return model


def drop_missing(rows: list[dict]) -> list[dict]:
    # stat calls on EFS are slow for large pools, so show progress
    kept = [r for r in progress(rows, len(rows), "Checking files exist") if Path(r["path"]).exists()]
    if len(kept) < len(rows):
        print(f"skipping {len(rows) - len(kept)} missing files")
    return kept


def review_sheet(rows: list[dict], out_path: Path, per_row: int = 8, thumb: int = 200) -> None:
    import cv2

    tiles = []
    for r in rows:
        img = cv2.imread(r["path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb / max(h, w)
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        canvas = np.zeros((thumb + 16, thumb, 3), np.uint8)
        canvas[16:16 + img.shape[0], :img.shape[1]] = img
        cv2.putText(canvas, f"{r['score']:.3f} {r['section']}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(canvas)
    if not tiles:
        return
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    rows_img = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
    cv2.imwrite(str(out_path), np.vstack(rows_img))


# -- main ---------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool_csv", type=Path, help="Candidate pool: path,label[,bbox][,group].")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target-label", type=int, choices=(0, 1), default=1, help="Class to mine (default 1 = replays).")
    ap.add_argument("--exclude-csv", type=Path, nargs="*", default=[], help="CSVs whose paths must not be mined (training set, eval sets).")
    ap.add_argument("--prod-csv", type=Path, help="Production samples of the target class for the production-likeness weight.")
    ap.add_argument("--prod-max", type=int, default=2000)
    ap.add_argument("--n-select", type=int, required=True)
    ap.add_argument("--band-low", type=float, default=0.6, help="Hardness percentile where the candidate band starts.")
    ap.add_argument("--band-high", type=float, default=0.95, help="Above this percentile: suspected label noise, held out for review.")
    ap.add_argument("--max-per-group", type=int, default=3)
    ap.add_argument("--group-col-from-path", action="store_true", help="Ignore a CSV group column and derive groups from paths.")
    ap.add_argument("--clusters", type=int, default=50)
    ap.add_argument("--max-cluster-frac", type=float, default=0.1)
    ap.add_argument("--base-csv", type=Path, help="Current training CSV; writes train_mix.csv = base + selected x repeat.")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--max-pool", type=int, default=0, help="Score at most this many pool rows (random subset).")
    ap.add_argument("--review-sheet", type=int, default=0, help="Thumbnails per section in review.jpg (0 = off).")
    ap.add_argument("--reuse", action="store_true", help="Reuse out-dir/embeddings.npz from an earlier run on the same pool.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--use-bbox-crop", action="store_true")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--resize-mode", choices=("squash", "letterbox"), default=None)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--mixed-precision", action="store_true")
    return ap.parse_args()


def main(args: argparse.Namespace | None = None, embed_fn=None) -> dict:
    args = args or parse_args()
    if not 0 <= args.band_low < args.band_high <= 1:
        raise SystemExit("need 0 <= --band-low < --band-high <= 1")
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    excluded = {r["orig"] for p in args.exclude_csv for r in read_rows(p)} | {r["path"] for p in args.exclude_csv for r in read_rows(p)}
    pool = [r for r in read_rows(args.pool_csv) if r["label"] == args.target_label]
    n_all = len(pool)
    pool = [r for r in pool if r["path"] not in excluded and r["orig"] not in excluded]
    print(f"pool: {n_all} rows of label {args.target_label}, {n_all - len(pool)} excluded, {len(pool)} left")
    if args.max_pool and len(pool) > args.max_pool:
        pool = [pool[i] for i in sorted(rng.choice(len(pool), args.max_pool, replace=False))]
    pool = drop_missing(pool) if embed_fn is None else pool
    if len(pool) < args.n_select:
        raise SystemExit(f"only {len(pool)} candidates for --n-select {args.n_select}")

    cache = args.out_dir / "embeddings.npz"
    model = None
    if args.reuse and cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["paths"]) != [r["path"] for r in pool]:
            raise SystemExit("--reuse: cached rows differ from this pool; delete embeddings.npz")
        feats, scores = z["feats"], z["scores"]
        print("reusing cached embeddings")
    else:
        if embed_fn is None:
            model = load_model_and_geometry(args)
            embed_fn = lambda rows, desc: compute_embeddings(rows, args, model, desc)
        feats, scores = embed_fn(pool, "Scoring pool")
        np.savez_compressed(cache, paths=np.array([r["path"] for r in pool]), feats=feats, scores=scores)

    labels = np.array([r["label"] for r in pool])
    hard = hardness_scores(scores, labels)
    pct = percentile_rank(hard)
    groups = [(r["group"] if r["group"] and not args.group_col_from_path else derive_group(r["path"])) for r in pool]

    weight = np.ones(len(pool))
    if args.prod_csv:
        if embed_fn is None:  # pool came from the cache
            model = load_model_and_geometry(args)
            embed_fn = lambda rows, desc: compute_embeddings(rows, args, model, desc)
        prod_rows = [r for r in read_rows(args.prod_csv) if r["label"] == args.target_label]
        if len(prod_rows) > args.prod_max:
            prod_rows = [prod_rows[i] for i in rng.choice(len(prod_rows), args.prod_max, replace=False)]
        prod_rows = drop_missing(prod_rows) if model is not None else prod_rows
        prod_feats, _ = embed_fn(prod_rows, "Embedding production reference")
        print("Fitting production-likeness probe", flush=True)
        weight = production_weights(feats, prod_feats, args.seed)

    in_band = (pct >= args.band_low) & (pct <= args.band_high)
    noise = pct > args.band_high
    cand = np.flatnonzero(in_band)
    if len(cand) < args.n_select:
        raise SystemExit(f"band holds {len(cand)} candidates, fewer than --n-select {args.n_select}; widen the band")
    cluster = np.full(len(pool), -1)
    print(f"Clustering {len(cand)} candidates", flush=True)
    cluster[cand] = assign_clusters(feats[cand], args.clusters, args.seed)
    ramp = 0.5 + 0.5 * (pct - args.band_low) / (args.band_high - args.band_low)
    sel_local = weighted_diverse_pick(
        (ramp * weight)[cand], [groups[i] for i in cand], cluster[cand], args.n_select,
        args.max_per_group, args.max_cluster_frac, rng)
    selected = sorted(int(cand[i]) for i in sel_local)
    if len(selected) < args.n_select:
        print(f"WARNING: caps allowed only {len(selected)} of {args.n_select}; raise --max-per-group or --max-cluster-frac")
    control = sorted(random_pick_grouped(groups, len(selected), args.max_per_group, rng))

    for i, r in enumerate(pool):
        r.update(score=float(scores[i]), hardness=float(hard[i]), pct=float(pct[i]), group=groups[i],
                 cluster=int(cluster[i]), prod_weight=float(weight[i]),
                 section="noise" if noise[i] else ("band" if in_band[i] else "easy"))
    write_csv(args.out_dir / "scored.csv", pool,
              ["path", "label", "bbox", "score", "hardness", "pct", "group", "cluster", "prod_weight", "section"])
    sel_rows = [pool[i] for i in selected]
    write_csv(args.out_dir / "selected.csv", sel_rows, ["path", "label", "bbox"])
    write_csv(args.out_dir / "control.csv", [pool[i] for i in control], ["path", "label", "bbox"])
    noise_rows = sorted((pool[i] for i in np.flatnonzero(noise)), key=lambda r: -r["hardness"])
    write_csv(args.out_dir / "review.csv", noise_rows, ["path", "label", "score", "hardness", "group"])

    if args.base_csv:
        base = read_rows(args.base_csv)
        mix = [{"path": r["orig"], "label": r["label"], "bbox": r["bbox"]} for r in base]
        mix += [{"path": r["path"], "label": r["label"], "bbox": r["bbox"]} for r in sel_rows] * args.repeat
        write_csv(args.out_dir / "train_mix.csv", mix, ["path", "label", "bbox"])

    if args.review_sheet:
        sample = (sel_rows[:args.review_sheet]
                  + noise_rows[:args.review_sheet])
        review_sheet(sample, args.out_dir / "review.jpg")

    sel_clusters = np.bincount(cluster[selected], minlength=max(cluster.max() + 1, 1))
    summary = {
        "pool_rows": len(pool), "target_label": args.target_label,
        "score_quantiles": {q: float(np.quantile(scores, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "band": [args.band_low, args.band_high], "candidates_in_band": int(len(cand)),
        "held_out_for_review": int(noise.sum()), "selected": len(selected), "control": len(control),
        "selected_groups": len({groups[i] for i in selected}),
        "largest_cluster_share": float(sel_clusters.max() / max(len(selected), 1)),
        "mean_score_selected": float(scores[selected].mean()), "mean_score_control": float(scores[control].mean()),
        "prod_weighting": bool(args.prod_csv), "base_repeat": args.repeat if args.base_csv else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
