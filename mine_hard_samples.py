#!/usr/bin/env python3
"""Pick hard, diverse replays from a scored pool to oversample in the next run.

Input is a CSV of model predictions you computed separately (predict_onnx_csv.py /
predict_checkpoint_csv.py): path,label,score[,bbox][,group][,dataset...]. Label
convention: 1 = replay (attack), 0 = live; score = P(label 1). No model or image
is loaded here, so it runs anywhere in seconds.

  hardness     score-error percentile within the class: 1 - score for replays,
               score for lives. Percentile, not an absolute threshold, because the
               model's EER threshold on production is nowhere near 0.5.
  band         only samples between --band-low and --band-high percentile are
               candidates. The very top slice (--band-high..1) is where mislabeled
               and unusable frames concentrate: it goes to review.csv, not to training.
  diversity    at most --max-per-group frames per session/video, and at most
               --max-bucket-frac of the selection from any bucket. A bucket is an
               embedding cluster with --embeddings, else the value of --bucket-col
               (e.g. dataset), else everything is one bucket and only the group cap applies.
  prod weight  needs embeddings: --embeddings (pool) and --prod-embeddings (production
               replays). A linear probe pool-vs-production weights replays that look
               like production over ones that are merely odd. Otherwise all weights are 1.

Optional embedding files are .npz: pool `paths` + `feats` (aligned by path to the
scores CSV, extra rows ignored, missing rows are an error); production `feats`.

Outputs in --out-dir:

  scored.csv    every candidate-class row: score, hardness, percentile, group, bucket, weight, section
  selected.csv  path,label,bbox -- the hard sample
  control.csv   same size, random from the same pool with the same per-group cap.
                TRAIN THIS ARM TOO: without it a gain cannot be told from "more data".
  review.csv    the suspected-noise top slice, plus review.jpg with --review-sheet
  train_mix.csv with --base-csv: base rows + selected rows x --repeat
  summary.json

Never mine rows the model trained on (their scores are biased) or any evaluation
set: pass them with --exclude-csv.

Example:

  python mine_hard_samples.py runs/scores/pool_scores.csv \\
      --exclude-csv data/train_current.csv data/ProdTest-0.3.csv \\
      --n-select 4000 --bucket-col dataset \\
      --base-csv data/train_current.csv --repeat 2 --out-dir runs/mining/r1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np



# -- pure selection logic -----------------------------------------------

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

def read_rows(csv_path: Path, need_score: bool = False) -> list[dict]:
    """path,label[,score][,bbox][,group] plus any extra columns, kept under their own names."""
    need = {"path", "label"} | ({"score"} if need_score else set())
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not need <= set(reader.fieldnames):
            raise SystemExit(f"{csv_path} must have columns {sorted(need)}")
        rows = []
        for r in reader:
            path = Path(r["path"].strip())
            if not path.is_absolute():
                path = csv_path.parent / path
            row = {k: (v or "").strip() for k, v in r.items() if k not in ("path", "label", "score", "bbox")}
            row.update(path=str(path), orig=r["path"].strip(), label=int(r["label"]), bbox=(r.get("bbox") or "").strip())
            if need_score:
                row["score"] = float(r["score"])
            rows.append(row)
    return rows


def load_embeddings(npz_path: Path, rows: list[dict]) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=True)
    index = {str(p): i for i, p in enumerate(z["paths"])}
    missing = [r["path"] for r in rows if r["path"] not in index and r["orig"] not in index]
    if missing:
        raise SystemExit(f"{npz_path}: {len(missing)} pool rows have no embedding, e.g. {missing[0]}")
    return z["feats"][[index.get(r["path"], index.get(r["orig"])) for r in rows]].astype(np.float32)


def label_buckets(rows: list[dict], column: str | None) -> np.ndarray:
    if not column:
        return np.zeros(len(rows), int)
    if column not in rows[0]:
        raise SystemExit(f"--bucket-col {column!r} is not a column of the scores CSV")
    codes: dict[str, int] = {}
    return np.array([codes.setdefault(r[column], len(codes)) for r in rows])


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


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
    ap.add_argument("scores_csv", type=Path, help="Predictions on the pool: path,label,score[,bbox][,group].")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target-label", type=int, choices=(0, 1), default=1, help="Class to mine (default 1 = replays).")
    ap.add_argument("--exclude-csv", type=Path, nargs="*", default=[], help="CSVs whose paths must not be mined (training set, eval sets).")
    ap.add_argument("--n-select", type=int, required=True)
    ap.add_argument("--band-low", type=float, default=0.6, help="Hardness percentile where the candidate band starts.")
    ap.add_argument("--band-high", type=float, default=0.95, help="Above this percentile: suspected label noise, held out for review.")
    ap.add_argument("--max-per-group", type=int, default=3)
    ap.add_argument("--group-col-from-path", action="store_true", help="Ignore a CSV group column and derive groups from paths.")
    ap.add_argument("--bucket-col", help="CSV column to cap the selection by (e.g. dataset). Ignored when --embeddings is given.")
    ap.add_argument("--max-bucket-frac", type=float, default=0.25, help="Max share of the selection from one --bucket-col value (0.1 is typical for embedding clusters).")
    ap.add_argument("--embeddings", type=Path, help=".npz with pool `paths` and `feats`: cluster buckets and production weight.")
    ap.add_argument("--prod-embeddings", type=Path, help=".npz with production replay `feats` for the production-likeness weight.")
    ap.add_argument("--prod-max", type=int, default=2000)
    ap.add_argument("--clusters", type=int, default=50, help="KMeans clusters when --embeddings is given.")
    ap.add_argument("--base-csv", type=Path, help="Current training CSV; writes train_mix.csv = base + selected x repeat.")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--review-sheet", type=int, default=0, help="Thumbnails per section in review.jpg (0 = off; needs the images and opencv).")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main(args: argparse.Namespace | None = None) -> dict:
    args = args or parse_args()
    if not 0 <= args.band_low < args.band_high <= 1:
        raise SystemExit("need 0 <= --band-low < --band-high <= 1")
    if args.prod_embeddings and not args.embeddings:
        raise SystemExit("--prod-embeddings needs --embeddings for the pool")
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    excluded = set()
    for p in args.exclude_csv:
        for r in read_rows(p):
            excluded.update((r["orig"], r["path"]))
    scored = read_rows(args.scores_csv, need_score=True)
    pool = [r for r in scored if r["label"] == args.target_label]
    n_class = len(pool)
    pool = [r for r in pool if r["path"] not in excluded and r["orig"] not in excluded]
    print(f"{len(scored)} scored rows, {n_class} of label {args.target_label}, {n_class - len(pool)} excluded, {len(pool)} candidates")
    if len(pool) < args.n_select:
        raise SystemExit(f"only {len(pool)} candidates for --n-select {args.n_select}")

    scores = np.array([r["score"] for r in pool])
    labels = np.array([r["label"] for r in pool])
    hard = hardness_scores(scores, labels)
    pct = percentile_rank(hard)
    groups = [(r.get("group") if r.get("group") and not args.group_col_from_path else derive_group(r["path"])) for r in pool]

    feats = load_embeddings(args.embeddings, pool) if args.embeddings else None
    weight = np.ones(len(pool))
    if args.prod_embeddings:
        prod = np.load(args.prod_embeddings, allow_pickle=True)["feats"].astype(np.float32)
        if len(prod) > args.prod_max:
            prod = prod[rng.choice(len(prod), args.prod_max, replace=False)]
        print("Fitting production-likeness probe", flush=True)
        weight = production_weights(feats, prod, args.seed)

    in_band = (pct >= args.band_low) & (pct <= args.band_high)
    noise = pct > args.band_high
    cand = np.flatnonzero(in_band)
    if len(cand) < args.n_select:
        raise SystemExit(f"band holds {len(cand)} candidates, fewer than --n-select {args.n_select}; widen the band")
    bucket = np.full(len(pool), -1)
    if feats is not None:
        print(f"Clustering {len(cand)} candidates", flush=True)
        bucket[cand] = assign_clusters(feats[cand], args.clusters, args.seed)
        bucket_frac = 0.1
    else:
        bucket[cand] = label_buckets([pool[i] for i in cand], args.bucket_col)
        bucket_frac = args.max_bucket_frac if args.bucket_col else 1.0
    ramp = 0.5 + 0.5 * (pct - args.band_low) / (args.band_high - args.band_low)
    sel_local = weighted_diverse_pick(
        (ramp * weight)[cand], [groups[i] for i in cand], bucket[cand], args.n_select,
        args.max_per_group, bucket_frac, rng)
    selected = sorted(int(cand[i]) for i in sel_local)
    if len(selected) < args.n_select:
        print(f"WARNING: caps allowed only {len(selected)} of {args.n_select}; raise --max-per-group or --max-bucket-frac")
    control = sorted(random_pick_grouped(groups, len(selected), args.max_per_group, rng))

    for i, r in enumerate(pool):
        r.update(hardness=float(hard[i]), pct=float(pct[i]), group=groups[i], bucket=int(bucket[i]),
                 prod_weight=float(weight[i]),
                 section="noise" if noise[i] else ("band" if in_band[i] else "easy"))
    write_csv(args.out_dir / "scored.csv", pool,
              ["path", "label", "bbox", "score", "hardness", "pct", "group", "bucket", "prod_weight", "section"])
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
        review_sheet(sel_rows[:args.review_sheet] + noise_rows[:args.review_sheet], args.out_dir / "review.jpg")

    sel_buckets = np.bincount(bucket[selected] - bucket[selected].min()) if selected else np.array([0])
    summary = {
        "pool_rows": len(pool), "target_label": args.target_label,
        "score_quantiles": {str(q): float(np.quantile(scores, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "band": [args.band_low, args.band_high], "candidates_in_band": int(len(cand)),
        "held_out_for_review": int(noise.sum()), "selected": len(selected), "control": len(control),
        "selected_groups": len({groups[i] for i in selected}),
        "largest_bucket_share": float(sel_buckets.max() / max(len(selected), 1)),
        "mean_score_selected": float(scores[selected].mean()), "mean_score_control": float(scores[control].mean()),
        "bucketing": "embedding clusters" if feats is not None else (args.bucket_col or "none"),
        "prod_weighting": bool(args.prod_embeddings), "base_repeat": args.repeat if args.base_csv else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
