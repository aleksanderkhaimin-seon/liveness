#!/usr/bin/env python3
"""Map every dataset into one embedding space and show how each relates to production.

Two steps, because only the first needs the images and TensorFlow:

  embed   (Code Editor, or anything that sees EFS) reads a registry of manifests
          (configs/domains.json), samples up to --max-per-cell rows per
          (domain, label) with at most --max-per-group frames per session, and writes
          one .npz per source to <out-dir>/emb/. A source already embedded with the
          same model and settings is skipped, so registering a new candidate dataset
          costs only its own embedding.
  report  (anywhere: numpy, scipy, scikit-learn; umap-learn if installed) reads
          those files and writes <out-dir>/domains.html, summary.json, samples.csv.

A domain is one value of a source's `split_col` (e.g. `dataset`), or the whole
source. Roles: prod (what the model has to work on), train (in the current
training set), test, candidate (not used yet: is it worth adding / collecting?).
Label 1 = attack, 0 = live.

Everything below runs on PCA-reduced, L2-normalised embeddings; neighbours from
the same session (folder holding the frames, or a `group` column) are skipped,
so near-duplicate frames do not count as agreement.

  coverage      per production sample: share of its k nearest same-label
                neighbours (production + train) that are train, over the share
                expected if the two were mixed evenly. ~1 = training data exists
                here; < --uncovered = production-only region, nothing to learn it from.
  prod lift     per non-prod (domain, label): how often it is among the k nearest
                non-prod neighbours of a production sample, over its share of the
                non-prod pool. > 1 = looks like production.
  auc vs prod   group-cross-validated linear probe domain vs production, same label
                (as in embedding_gap.py: 0.5 = indistinguishable, 1.0 = separable).
  Δ covered     train domain: production samples that become uncovered without it.
                test / candidate domain: production samples that become covered
                if it were added to train. The number to rank collection by.
  affinity      neighbour-share lift between every pair of (domain, label) cells.
  clusters      KMeans regions labelled `production gap` (production with little
                train data), `pool only` (no production) or `shared`.

Passing --scores (path,score CSVs, e.g. *_predictions.csv from launch_eval.py) or
having a `score` column in a manifest adds model error at the production EER
threshold per region, which checks that low coverage is where the model fails.

The embedding is the trained model's pooled features (--checkpoint: what the
current model separates) or ImageNet EfficientNetB2 (--imagenet: plain visual
similarity, independent of what the current training data taught). Each output
directory holds one embedding; the report refuses to mix two.

Thumbnails (--thumb-size, off by default) put small whole-frame JPEGs of every
source, production included, into the .npz and a few per cluster into the HTML.
Set `"thumbs": false` on a source to leave it out.

The .npz files carry `paths` and `feats`, so they also work as --embeddings /
--prod-embeddings for mine_hard_samples.py.

Example (Code Editor):

  python dataset_domains.py embed --config configs/domains.json \\
      --checkpoint eval_results/<job>/output/best.keras --out-dir runs/domains/m1
  python dataset_domains.py report --config configs/domains.json \\
      --out-dir runs/domains/m1 --scores eval_results/<eval-job>/output/m1_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROLES = ("prod", "train", "test", "candidate")
LABEL_NAMES = {0: "live", 1: "attack"}
TEMPLATE = Path(__file__).with_name("dataset_domains_template.html")


# -- registry -------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    """{"path_map": {old_prefix: new_prefix}, "sources": [...], "domain_tags": {domain: {k: v}}}."""
    cfg = json.loads(path.read_text(encoding="utf-8"))
    names = set()
    for src in cfg.get("sources", []):
        src.setdefault("name", Path(src["csv"]).stem)
        src.setdefault("role", "train")
        if src["role"] not in ROLES:
            raise SystemExit(f"source {src['name']}: role must be one of {ROLES}")
        if src["name"] in names:
            raise SystemExit(f"duplicate source name {src['name']!r}")
        names.add(src["name"])
        meta = src.get("meta_cols") or {}
        src["meta_cols"] = {c: c for c in meta} if isinstance(meta, list) else dict(meta)
        src.setdefault("tags", {})
    if not names:
        raise SystemExit(f"{path} lists no sources")
    cfg.setdefault("path_map", {})
    cfg.setdefault("domain_tags", {})
    return cfg


def session_of(path: str, group_by: str) -> str:
    """What counts as one capture. auto: the video for `.../<video>/frames/x.jpg`, else the file itself
    (still images share folders like `images/` without being related); parent: the folder holding the file."""
    p = Path(path)
    if group_by == "parent":
        return str(p.parent)
    if group_by == "file":
        return path
    return str(p.parent.parent) if p.parent.name.lower() == "frames" else path


def remap(path: str, path_map: dict[str, str]) -> str:
    for old, new in path_map.items():
        old = old.rstrip("/")
        if path == old or path.startswith(old + "/"):
            return new.rstrip("/") + path[len(old):]
    return path


def read_manifest(src: dict, path_map: dict[str, str]) -> list[dict]:
    csv_path = Path(src["csv"])
    split_col = src.get("split_col")
    need = {"path", "label"} | ({split_col} if split_col else set())
    rows, seen = [], set()
    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not need <= set(reader.fieldnames):
            raise SystemExit(f"{csv_path} must have columns {sorted(need)}")
        for r in reader:
            orig = (r["path"] or "").strip()
            if not orig or orig in seen:
                continue
            seen.add(orig)
            label = int(float(r["label"]))
            if label not in (0, 1):
                raise SystemExit(f"{csv_path}: label must be 0 or 1, got {r['label']!r}")
            path = remap(orig, path_map)
            if not Path(path).is_absolute():
                path = str(csv_path.parent / path)
            score = (r.get("score") or "").strip()
            rows.append({
                "path": path, "orig": orig, "label": label,
                "bbox": (r.get("bbox") or "").strip(),
                "domain": ((r.get(split_col) or "").strip() or "(empty)") if split_col else src["name"],
                "group": (r.get("group") or "").strip() or session_of(path, src.get("group_by", "auto")),
                "score": float(score) if score else math.nan,
                "meta": {out: (r.get(col) or "").strip() for col, out in src["meta_cols"].items()},
            })
    return rows


# -- embed --------------------------------------------------------------------------------

def sample_cells(rows: list[dict], max_per_cell: int, max_per_group: int, need_bbox: bool,
                 rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    """Random rows per (domain, label), session-capped; existence is checked only for rows considered."""
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["domain"], r["label"])].append(r)
    picked, stats = [], []
    for (domain, label), members in sorted(by_cell.items()):
        per_group: Counter = Counter()
        kept, missing, no_bbox = [], 0, 0
        for i in rng.permutation(len(members)):
            r = members[i]
            if per_group[r["group"]] >= max_per_group:
                continue
            if need_bbox and not r["bbox"]:
                no_bbox += 1
                continue
            if not os.path.exists(r["path"]):
                missing += 1
                if missing >= 200 and not kept:
                    print(f"  WARNING: {domain}/{LABEL_NAMES[label]}: first 200 files missing "
                          f"(e.g. {r['path']}); check path_map. Skipping the cell.")
                    break
                continue
            per_group[r["group"]] += 1
            kept.append(r)
            if len(kept) >= max_per_cell:
                break
        stats.append({"domain": domain, "label": label, "rows": len(members),
                      "groups": len({m["group"] for m in members}), "sampled": len(kept),
                      "missing_seen": missing, "no_bbox": no_bbox})
        picked.extend(kept)
    return picked, stats


def build_extractor(args):
    """Pooled-feature model plus a description of it; sets the training pipeline's input geometry."""
    import tensorflow as tf

    from input_geometry import find_report, resolve_geometry
    from train_efficientnet_b2 import configure_runtime, set_input_geometry

    configure_runtime(args.require_gpu, args.mixed_precision)
    if args.imagenet:
        geo = resolve_geometry(args.image_size, args.resize_mode)
        set_input_geometry(geo.image_size, geo.resize_mode)
        extractor = tf.keras.applications.EfficientNetB2(
            include_top=False, weights="imagenet", pooling="avg", input_shape=(geo.image_size, geo.image_size, 3))
        model_id = "imagenet:EfficientNetB2"
    else:
        from predict_checkpoint_csv import load_model

        report = find_report(args.checkpoint)
        geo = resolve_geometry(args.image_size, args.resize_mode, report_path=report)
        set_input_geometry(geo.image_size, geo.resize_mode)
        model = load_model(args.checkpoint)
        geo = resolve_geometry(args.image_size, args.resize_mode, model_input_size=int(model.input_shape[1]), report_path=report)
        set_input_geometry(geo.image_size, geo.resize_mode)
        # Feature input of the last Dense layer == pooled backbone output (Dropout is identity at inference).
        extractor = tf.keras.Model(model.input, model.get_layer("live_score").input)
        stat = args.checkpoint.stat()
        model_id = f"checkpoint:{args.checkpoint.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"
    print(f"Input geometry: {geo}")
    return extractor, {"model": model_id, "image_size": geo.image_size, "resize_mode": geo.resize_mode,
                       "use_bbox_crop": args.use_bbox_crop, "margin": args.margin}


def embed_rows(extractor, rows: list[dict], args, desc: str) -> np.ndarray:
    from progress_util import progress
    from train_efficientnet_b2 import make_dataset

    dataset = make_dataset([r["path"] for r in rows], [r["label"] for r in rows], [r["bbox"] for r in rows],
                           args.batch_size, training=False, use_bbox_crop=args.use_bbox_crop, margin=args.margin)
    total = math.ceil(len(rows) / args.batch_size)
    chunks = [extractor(batch[0], training=False).numpy() for batch in progress(dataset, total, desc)]
    return np.concatenate(chunks).astype(np.float32)


def make_thumbnails(rows: list[dict], size: int) -> tuple[np.ndarray, np.ndarray]:
    """Whole-frame JPEGs, long side `size`, as one byte buffer + offsets (no pickled objects in the .npz)."""
    from PIL import Image

    data, offsets = bytearray(), [0]
    for r in rows:
        try:
            with Image.open(r["path"]) as im:
                im.draft("RGB", (size * 2, size * 2))
                im = im.convert("RGB")
                im.thumbnail((size, size))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=70)
                data += buf.getvalue()
        except Exception as error:  # an unreadable thumbnail must not lose the embedding
            print(f"  thumbnail failed for {r['path']}: {error}")
        offsets.append(len(data))
    return np.frombuffer(bytes(data), np.uint8), np.asarray(offsets, np.int64)


def source_signature(src: dict, cfg: dict, embedding: dict, args) -> dict:
    stat = Path(src["csv"]).stat()
    return {
        "embedding": embedding, "csv": str(Path(src["csv"]).resolve()), "csv_size": stat.st_size,
        "csv_mtime": int(stat.st_mtime), "split_col": src.get("split_col"), "meta_cols": src["meta_cols"],
        "group_by": src.get("group_by", "auto"),
        "path_map": cfg["path_map"], "max_per_cell": args.max_per_cell, "max_per_group": args.max_per_group,
        "seed": args.seed, "thumb_size": args.thumb_size if src.get("thumbs", True) else 0,
    }


def read_info(npz_path: Path) -> dict | None:
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            return json.loads(str(z["info"]))
    except Exception:
        return None


def run_embed(args: argparse.Namespace) -> None:
    if bool(args.checkpoint) == bool(args.imagenet):
        raise SystemExit("pass exactly one of --checkpoint or --imagenet")
    cfg = load_config(args.config)
    only = set(args.only or [])
    emb_dir = args.out_dir / "emb"
    emb_dir.mkdir(parents=True, exist_ok=True)
    extractor, embedding = build_extractor(args)

    for src in cfg["sources"]:
        if only and src["name"] not in only:
            continue
        out = emb_dir / f"{src['name']}.npz"
        signature = source_signature(src, cfg, embedding, args)
        old = read_info(out) if out.exists() else None
        if old and old.get("signature") == signature and not args.force:
            print(f"[{src['name']}] up to date, skipping")
            continue
        if old:
            changed = sorted(k for k in signature if old.get("signature", {}).get(k) != signature[k])
            print(f"[{src['name']}] re-embedding, changed: {', '.join(changed) or 'forced'}")
        rows = read_manifest(src, cfg["path_map"])
        rng = np.random.default_rng(args.seed)
        picked, stats = sample_cells(rows, args.max_per_cell, args.max_per_group, args.use_bbox_crop, rng)
        print(f"[{src['name']}] {len(rows)} rows, {len(stats)} cells, {len(picked)} sampled")
        for s in stats:
            print(f"    {s['domain']:<40} {LABEL_NAMES[s['label']]:<6} rows {s['rows']:>7}  sampled {s['sampled']:>5}"
                  + (f"  missing {s['missing_seen']}" if s["missing_seen"] else ""))
        if not picked:
            print(f"[{src['name']}] nothing to embed")
            continue
        started = time.time()
        feats = embed_rows(extractor, picked, args, f"Embedding {src['name']} ({len(picked)} images)")
        arrays = {
            "paths": np.array([r["path"] for r in picked]), "orig": np.array([r["orig"] for r in picked]),
            "labels": np.array([r["label"] for r in picked], np.int8),
            "domains": np.array([r["domain"] for r in picked]), "groups": np.array([r["group"] for r in picked]),
            "scores": np.array([r["score"] for r in picked], np.float32), "feats": feats,
        }
        for out_name in src["meta_cols"].values():
            arrays[f"meta_{out_name}"] = np.array([r["meta"][out_name] for r in picked])
        if signature["thumb_size"]:
            arrays["thumb_data"], arrays["thumb_offsets"] = make_thumbnails(picked, signature["thumb_size"])
        info = {"source": src["name"], "signature": signature, "cells": stats,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - started, 1)}
        np.savez_compressed(out, info=np.array(json.dumps(info)), **arrays)
        print(f"[{src['name']}] wrote {out}")


# -- report: loading ------------------------------------------------------------------------

def load_embedded(cfg: dict, emb_dir: Path, score_csvs: list[Path]) -> dict:
    parts, infos, embeddings = [], {}, set()
    for src in cfg["sources"]:
        path = emb_dir / f"{src['name']}.npz"
        if not path.exists():
            print(f"WARNING: {path} missing; run embed for {src['name']}. Leaving it out.")
            continue
        with np.load(path, allow_pickle=False) as z:
            info = json.loads(str(z["info"]))
            part = {k: z[k] for k in z.files if k != "info"}
        embeddings.add(json.dumps(info["signature"]["embedding"], sort_keys=True))
        part["source"] = src
        infos[src["name"]] = info
        parts.append(part)
    if not parts:
        raise SystemExit(f"no embeddings under {emb_dir}")
    if len(embeddings) > 1:
        raise SystemExit("sources in this directory were embedded with different models or input settings:\n  "
                         + "\n  ".join(sorted(embeddings)) + "\nre-run embed (with --force) or use another --out-dir")

    # A domain name produced by two sources is ambiguous: prefix it with the source.
    producers: dict[str, set] = defaultdict(set)
    for p in parts:
        for d in np.unique(p["domains"]):
            producers[str(d)].add(p["source"]["name"])

    cols = defaultdict(list)
    seen: dict[str, str] = {}
    overlaps: Counter = Counter()
    thumbs: list[bytes | None] = []
    meta_names = sorted({k[5:] for p in parts for k in p if k.startswith("meta_")})
    for p in parts:
        src = p["source"]
        n = len(p["paths"])
        keep = []
        for i in range(n):
            key = str(p["orig"][i])
            if key in seen:
                overlaps[(seen[key], src["name"])] += 1
                continue
            seen[key] = src["name"]
            keep.append(i)
        keep = np.asarray(keep, int)
        domains = [str(d) if len(producers[str(d)]) == 1 else f"{src['name']}/{d}" for d in p["domains"][keep]]
        cols["domain"] += domains
        cols["source"] += [src["name"]] * len(keep)
        cols["role"] += [src["role"]] * len(keep)
        for k in ("paths", "orig", "labels", "groups", "scores", "feats"):
            cols[k].append(p[k][keep])
        for m in meta_names:
            cols[f"meta_{m}"] += [str(v) for v in p[f"meta_{m}"][keep]] if f"meta_{m}" in p else [""] * len(keep)
        if "thumb_offsets" in p:
            off, data = p["thumb_offsets"], p["thumb_data"]
            thumbs += [bytes(data[off[i]:off[i + 1]]) or None for i in keep]
        else:
            thumbs += [None] * len(keep)
    data = {k: (np.concatenate(v) if isinstance(v[0], np.ndarray) else np.asarray(v)) for k, v in cols.items()}
    data["labels"] = data["labels"].astype(int)
    data["scores"] = data["scores"].astype(np.float64)
    data["thumbs"] = thumbs
    data["meta_names"] = meta_names
    data["infos"] = infos
    data["embedding"] = json.loads(next(iter(embeddings)))
    data["overlaps"] = [{"kept_in": a, "dropped_from": b, "rows": c} for (a, b), c in overlaps.items()]

    index = {}
    for i, (p, o) in enumerate(zip(data["paths"], data["orig"])):
        index[str(p)] = index[str(o)] = i
    for csv_path in score_csvs:
        hits = 0
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                i = index.get((r.get("path") or "").strip())
                if i is not None and (r.get("score") or "").strip():
                    data["scores"][i] = float(r["score"])
                    hits += 1
        print(f"{csv_path}: scores for {hits} embedded samples")
    return data


# -- report: measurements --------------------------------------------------------------------

def reduce(feats: np.ndarray, cell: np.ndarray, dims: int, seed: int) -> tuple[np.ndarray, float]:
    """PCA fitted on a cell-balanced subset (no domain dominates the axes), then L2 norm: euclidean ~ cosine."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    fit = np.concatenate([rng.permutation(np.flatnonzero(cell == c))[:1000] for c in np.unique(cell)])
    pca = PCA(n_components=min(dims, len(fit) - 1, feats.shape[1]), random_state=seed).fit(feats[fit])
    z = pca.transform(feats)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    return z.astype(np.float32), float(pca.explained_variance_ratio_.sum())


def neighbours(index_z: np.ndarray, index_group: np.ndarray, query_z: np.ndarray, query_group: np.ndarray,
               k: int) -> np.ndarray:
    """k nearest rows of the index for each query, skipping the query's own session. -1 pads short rows."""
    from sklearn.neighbors import NearestNeighbors

    out = np.full((len(query_z), k), -1, int)
    if len(index_z) == 0 or len(query_z) == 0:
        return out
    nn = NearestNeighbors(n_neighbors=min(k + 30, len(index_z))).fit(index_z)
    idx = nn.kneighbors(query_z, return_distance=False)
    for i, row in enumerate(idx):
        keep = row[index_group[row] != query_group[i]][:k]
        out[i, :len(keep)] = keep
    return out


def coverage_of(z, groups, prod_idx, pool_idx, k) -> np.ndarray:
    """Per production sample: pool share of its k nearest (prod ∪ pool) neighbours over the chance share."""
    if len(pool_idx) == 0:
        return np.zeros(len(prod_idx))
    union = np.r_[prod_idx, pool_idx]
    is_pool = np.r_[np.zeros(len(prod_idx), bool), np.ones(len(pool_idx), bool)]
    nb = neighbours(z[union], groups[union], z[prod_idx], groups[prod_idx], k)
    valid = nb >= 0
    frac = np.where(valid, is_pool[np.maximum(nb, 0)], False).sum(1) / np.maximum(valid.sum(1), 1)
    chance = len(pool_idx) / (len(pool_idx) + len(prod_idx) - 1)
    return frac / chance


def probe_auc(za: np.ndarray, ga: np.ndarray, zb: np.ndarray, gb: np.ndarray, seed: int) -> float | None:
    """Cross-validated AUC of a linear probe a-vs-b, folds split by session so near-duplicates never straddle them."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

    if min(len(za), len(zb)) < 20:
        return None
    x = np.vstack([za, zb])
    y = np.r_[np.zeros(len(za)), np.ones(len(zb))]
    g = np.r_[ga, gb]
    folds = 5
    if min(len(np.unique(ga)), len(np.unique(gb))) >= folds:
        cv, kw = StratifiedGroupKFold(folds, shuffle=True, random_state=seed), {"groups": g}
    else:
        cv, kw = StratifiedKFold(folds, shuffle=True, random_state=seed), {}
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    p = cross_val_predict(clf, x, y, cv=cv, method="predict_proba", **kw)[:, 1]
    return float(roc_auc_score(y, p))


def eer_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Threshold where attacks missed (score < t) equals lives rejected (score >= t); (eer, threshold)."""
    att, live = np.sort(scores[labels == 1]), np.sort(scores[labels == 0])
    t = np.unique(scores)
    miss = np.searchsorted(att, t, side="left") / len(att)
    rejected = 1 - np.searchsorted(live, t, side="left") / len(live)
    i = int(np.argmin(np.abs(miss - rejected)))
    return float((miss[i] + rejected[i]) / 2), float(t[i])


def project(z: np.ndarray, method: str, seed: int) -> tuple[np.ndarray, str]:
    if method in ("auto", "umap"):
        try:
            import umap  # type: ignore

            return umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=seed).fit_transform(z), "UMAP"
        except ImportError:
            if method == "umap":
                raise SystemExit("--projection umap needs `pip install umap-learn`")
    if method in ("auto", "tsne"):
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=min(30, len(z) // 4),
                    random_state=seed).fit_transform(z), "t-SNE"
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=seed).fit_transform(z), "PCA"


def map_subset(cell: np.ndarray, is_prod: np.ndarray, cap_total: int, rng) -> np.ndarray:
    """All production plus an equal per-cell cap for the rest, so small domains stay visible."""
    prod = np.flatnonzero(is_prod)
    rest_cells = {c: np.flatnonzero((cell == c) & ~is_prod) for c in np.unique(cell[~is_prod])}
    budget = max(cap_total - len(prod), 0)
    lo, hi = 0, max((len(v) for v in rest_cells.values()), default=0)
    while lo < hi:  # largest per-cell cap that fits the budget
        mid = (lo + hi + 1) // 2
        if sum(min(len(v), mid) for v in rest_cells.values()) <= budget:
            lo = mid
        else:
            hi = mid - 1
    picked = [prod] + [rng.permutation(v)[:lo] for v in rest_cells.values()]
    return np.sort(np.concatenate(picked)).astype(int)


def r3(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 3)


def analyse(data: dict, cfg: dict, args) -> dict:
    rng = np.random.default_rng(args.seed)
    labels, roles, domains = data["labels"], data["role"], data["domain"]
    group_codes = np.unique(data["groups"], return_inverse=True)[1]
    cell_names = sorted({(d, int(l)) for d, l in zip(domains, labels)},
                        key=lambda c: (ROLES.index(roles[np.flatnonzero(domains == c[0])[0]]), c[0], c[1]))
    cell_index = {c: i for i, c in enumerate(cell_names)}
    cell = np.array([cell_index[(d, int(l))] for d, l in zip(domains, labels)])
    is_prod = roles == "prod"
    if not is_prod.any():
        raise SystemExit("no source with role prod was embedded; coverage needs production")
    n = len(labels)
    print(f"{n} samples, {len(cell_names)} (domain, label) cells; reducing to {args.dims} dims")
    z, explained = reduce(data["feats"], cell, args.dims, args.seed)
    k = args.k

    # Production coverage against train, per label.
    coverage = np.full(n, np.nan)
    for lab in (0, 1):
        prod_idx = np.flatnonzero(is_prod & (labels == lab))
        train_idx = np.flatnonzero((roles == "train") & (labels == lab))
        if len(prod_idx):
            coverage[prod_idx] = coverage_of(z, group_codes, prod_idx, train_idx, k)
    uncovered = is_prod & (coverage < args.uncovered)

    # Which non-prod domains production's neighbours come from (prod lift) and the probe AUC.
    nb_domain_share: dict[int, Counter] = {}
    cell_stats = {i: {} for i in range(len(cell_names))}
    prod_nearest = [None] * n
    for lab in (0, 1):
        prod_idx = np.flatnonzero(is_prod & (labels == lab))
        pool_idx = np.flatnonzero(~is_prod & (labels == lab))
        if not len(prod_idx) or not len(pool_idx):
            continue
        nb = neighbours(z[pool_idx], group_codes[pool_idx], z[prod_idx], group_codes[prod_idx], k)
        hits = Counter(cell[pool_idx[j]] for j in nb.ravel() if j >= 0)
        total_hits = sum(hits.values())
        pool_cells = Counter(cell[pool_idx])
        for c, cnt in pool_cells.items():
            cell_stats[c]["prod_lift"] = (hits[c] / total_hits) / (cnt / len(pool_idx)) if total_hits else None
        for qi, row in zip(prod_idx, nb):
            prod_nearest[qi] = Counter(domains[pool_idx[j]] for j in row if j >= 0).most_common(1)[0][0] \
                if (row >= 0).any() else None
        nb_domain_share[lab] = hits
        sub_prod = rng.permutation(prod_idx)[:1500]
        for c in pool_cells:
            members = np.flatnonzero(cell == c)
            cell_stats[c]["auc_vs_prod"] = probe_auc(z[members], group_codes[members], z[sub_prod],
                                                     group_codes[sub_prod], args.seed)

    # Δ covered: train domains by leave-one-out, other non-prod domains by adding them to train.
    for c, (dom, lab) in enumerate(cell_names):
        role = roles[cell == c][0]
        if role == "prod":
            continue
        prod_idx = np.flatnonzero(is_prod & (labels == lab))
        if not len(prod_idx):
            continue
        train_idx = np.flatnonzero((roles == "train") & (labels == lab))
        base = coverage[prod_idx] >= args.uncovered
        if role == "train":
            alt = coverage_of(z, group_codes, prod_idx, train_idx[cell[train_idx] != c], k) >= args.uncovered
            cell_stats[c]["delta_covered"] = -int((base & ~alt).sum())
        else:
            alt = coverage_of(z, group_codes, prod_idx, np.r_[train_idx, np.flatnonzero(cell == c)], k) >= args.uncovered
            cell_stats[c]["delta_covered"] = int((alt & ~base).sum())

    # Affinity: neighbour-share lift between cells, over all samples.
    print("Neighbour graph for the affinity matrix")
    nb_all = neighbours(z, group_codes, z, group_codes, k)
    n_cells = len(cell_names)
    counts = np.zeros((n_cells, n_cells))
    src_cell, dst = np.repeat(cell, k), nb_all.ravel()
    np.add.at(counts, (src_cell[dst >= 0], cell[dst[dst >= 0]]), 1)
    sizes = np.bincount(cell, minlength=n_cells).astype(float)
    expected = counts.sum(1, keepdims=True) * sizes[None, :] / n
    log_lift = np.log2((counts + 1) / (expected + 1))
    order = list(range(n_cells))
    if n_cells > 2:
        from scipy.cluster.hierarchy import leaves_list, linkage

        sym = (log_lift + log_lift.T) / 2
        order = [int(i) for i in leaves_list(linkage(sym, method="average", metric="correlation"))]

    # Regions.
    from sklearn.cluster import KMeans

    n_clusters = max(2, min(args.clusters, n // 20))
    print(f"KMeans with {n_clusters} clusters")
    km = KMeans(n_clusters=n_clusters, n_init=4, random_state=args.seed).fit(z)
    cluster = km.labels_

    # Model error at the production EER threshold.
    scores = data["scores"]
    has_score = np.isfinite(scores)
    prod_scored = is_prod & has_score
    model = None
    error = np.full(n, np.nan)
    if prod_scored.any() and len(np.unique(labels[prod_scored])) == 2:
        eer, thr = eer_threshold(scores[prod_scored], labels[prod_scored])
        error[has_score] = np.where(labels[has_score] == 1, scores[has_score] < thr, scores[has_score] >= thr)
        model = {"prod_eer": eer, "threshold": thr, "prod_scored": int(prod_scored.sum())}
        for lab in (0, 1):
            sel = prod_scored & (labels == lab)
            unc, cov = sel & uncovered, sel & ~uncovered
            model[f"err_uncovered_{lab}"] = float(error[unc].mean()) if unc.any() else None
            model[f"err_covered_{lab}"] = float(error[cov].mean()) if cov.any() else None
            model[f"n_uncovered_{lab}"] = int(unc.sum())
            model[f"n_covered_{lab}"] = int(cov.sum())

    # Projection.
    sub = map_subset(cell, is_prod, args.map_max, rng)
    print(f"Projecting {len(sub)} points")
    xy, projection = project(z[sub], args.projection, args.seed)
    xy = (xy - xy.min(0)) / np.maximum(xy.max(0) - xy.min(0), 1e-9)

    return {"z": z, "explained": explained, "cell_names": cell_names, "cell": cell, "cell_stats": cell_stats,
            "coverage": coverage, "uncovered": uncovered, "prod_nearest": prod_nearest, "log_lift": log_lift,
            "order": order, "cluster": cluster, "centroids": km.cluster_centers_, "error": error, "model": model,
            "sub": sub, "xy": xy, "projection": projection, "nb_domain_share": nb_domain_share}


# -- report: assembling ----------------------------------------------------------------------

def domain_rows(data: dict, res: dict, cfg: dict) -> list[dict]:
    srcs = {s["name"]: s for s in cfg["sources"]}
    full_rows = {(name, c["domain"], c["label"]): c for name, info in data["infos"].items() for c in info["cells"]}
    out = {}
    for ci, (dom, lab) in enumerate(res["cell_names"]):
        members = np.flatnonzero(res["cell"] == ci)
        source = data["source"][members[0]]
        raw_dom = dom.split("/", 1)[1] if dom.startswith(source + "/") else dom
        entry = out.setdefault(dom, {
            "domain": dom, "source": source, "role": data["role"][members[0]],
            "tags": {**srcs[source]["tags"], **cfg["domain_tags"].get(raw_dom, {})}, "cells": {}})
        full = full_rows.get((source, raw_dom, lab), {})
        st = res["cell_stats"][ci]
        entry["cells"][LABEL_NAMES[lab]] = {
            "cell": ci, "rows": full.get("rows"), "groups": full.get("groups"), "sampled": int(len(members)),
            "prod_lift": r3(st.get("prod_lift")), "auc_vs_prod": r3(st.get("auc_vs_prod")),
            "delta_covered": st.get("delta_covered"), "self_lift": r3(res["log_lift"][ci, ci]),
            "error": r3(np.nanmean(res["error"][members])) if np.isfinite(res["error"][members]).any() else None,
            "nearest": [res["cell_names"][j][0] + " · " + LABEL_NAMES[res["cell_names"][j][1]]
                        for j in np.argsort(-res["log_lift"][ci]) if j != ci][:3],
        }
    return list(out.values())


def prod_breakdown(data: dict, res: dict, min_support: int) -> list[dict]:
    is_prod = data["role"] == "prod"
    out = []
    for m in data["meta_names"]:
        values = data[f"meta_{m}"]
        for lab in (0, 1):
            sel = is_prod & (data["labels"] == lab)
            for v, cnt in Counter(values[sel]).most_common():
                if cnt < min_support or not v:
                    continue
                idx = np.flatnonzero(sel & (values == v))
                err = res["error"][idx]
                near = Counter(res["prod_nearest"][i] for i in idx if res["prod_nearest"][i]).most_common(2)
                out.append({"column": m, "value": v, "label": LABEL_NAMES[lab], "n": int(cnt),
                            "coverage": r3(np.minimum(res["coverage"][idx], 1).mean()),
                            "uncovered": int(res["uncovered"][idx].sum()),
                            "error": r3(np.nanmean(err)) if np.isfinite(err).any() else None,
                            "nearest": [f"{d} ({c})" for d, c in near]})
    out.sort(key=lambda r: (-r["uncovered"], -r["n"]))
    return out


def thumb_uri(blob: bytes | None) -> str | None:
    import base64

    return "data:image/jpeg;base64," + base64.b64encode(blob).decode() if blob else None


def cluster_rows(data: dict, res: dict, args) -> list[dict]:
    roles, labels, domains = data["role"], data["labels"], data["domain"]
    is_prod = roles == "prod"
    prod_share = is_prod.mean()
    pos = {int(i): j for j, i in enumerate(res["sub"])}
    out = []
    for c in range(len(res["centroids"])):
        members = np.flatnonzero(res["cluster"] == c)
        prod_m = members[is_prod[members]]
        cov = np.minimum(res["coverage"][prod_m], 1).mean() if len(prod_m) else None
        lift = (len(prod_m) / len(members)) / prod_share if len(members) else 0
        status = "pool only" if lift < 0.2 else ("production gap" if cov is not None and cov < args.uncovered else "shared")
        meta = []
        for m in data["meta_names"]:
            meta += [f"{m}={v} ({k})" for v, k in Counter(data[f"meta_{m}"][prod_m]).most_common(3) if v]
        err = res["error"][prod_m]
        on_map = [pos[i] for i in members if i in pos]
        centre = res["xy"][on_map].mean(0) if on_map else None
        thumbs = []
        if args.thumbs_per_cluster:
            # Closest to the centroid first: up to half production, the rest from everything else.
            dist = ((res["z"][members] - res["centroids"][c]) ** 2).sum(1)
            ranked = [int(i) for i in members[np.argsort(dist)] if data["thumbs"][i]]
            prod_t = [i for i in ranked if is_prod[i]][:args.thumbs_per_cluster // 2]
            pool_t = [i for i in ranked if not is_prod[i]][:args.thumbs_per_cluster - len(prod_t)]
            thumbs = [{"src": thumb_uri(data["thumbs"][i]), "role": str(roles[i]),
                       "caption": f"{domains[i]} · {LABEL_NAMES[int(labels[i])]}"} for i in prod_t + pool_t]
        out.append({
            "id": c, "n": int(len(members)), "status": status, "prod_lift": r3(lift),
            "roles": {r: int((roles[members] == r).sum()) for r in ROLES},
            "labels": {LABEL_NAMES[l]: int((labels[members] == l).sum()) for l in (0, 1)},
            "domains": [f"{d} ({k})" for d, k in Counter(domains[members]).most_common(4)],
            "prod_meta": meta, "coverage": r3(cov), "prod_n": int(len(prod_m)),
            "error": r3(np.nanmean(err)) if np.isfinite(err).any() else None,
            "x": r3(centre[0]) if centre is not None else None, "y": r3(centre[1]) if centre is not None else None,
            "thumbs": thumbs,
        })
    order = {"production gap": 0, "shared": 1, "pool only": 2}
    out.sort(key=lambda r: (order[r["status"]], -r["prod_n"]))
    return out


def findings(data: dict, res: dict, doms: list[dict], breakdown: list[dict], args) -> list[dict]:
    out = []
    is_prod = data["role"] == "prod"
    for lab in (1, 0):
        sel = is_prod & (data["labels"] == lab)
        if not sel.any():
            continue
        unc = res["uncovered"][sel].mean()
        out.append({"kind": "coverage", "text":
                    f"{100 * unc:.0f}% of production {LABEL_NAMES[lab]}s ({res['uncovered'][sel].sum()} of {sel.sum()}) "
                    f"sit where train has almost no data (coverage < {args.uncovered})."})
    m = res["model"]
    if m:
        for lab in (1, 0):
            a, b = m.get(f"err_uncovered_{lab}"), m.get(f"err_covered_{lab}")
            if a is not None and b is not None and min(m[f"n_uncovered_{lab}"], m[f"n_covered_{lab}"]) >= 10:
                what = "missed" if lab == 1 else "rejected"
                out.append({"kind": "model", "text":
                            f"At the production EER threshold, uncovered production {LABEL_NAMES[lab]}s are {what} "
                            f"{100 * a:.0f}% of the time vs {100 * b:.0f}% for covered ones "
                            f"({'coverage tracks model failure' if a > b + 0.05 else 'coverage does not explain the errors here'})."})
    gaps = [r for r in breakdown if r["uncovered"] >= max(5, 0.5 * r["n"])][:5]
    if gaps:
        out.append({"kind": "collect", "text": "Least covered production groups: " + "; ".join(
            f"{r['value']} {r['label']}s ({r['uncovered']}/{r['n']} uncovered)" for r in gaps) + "."})
    train_rows = {lab: [c["rows"] for d in doms if d["role"] == "train" for l, c in d["cells"].items()
                        if l == lab and c["rows"]] for lab in LABEL_NAMES.values()}
    for d in doms:
        for lab, c in d["cells"].items():
            if d["role"] in ("candidate", "test") and (c["delta_covered"] or 0) > 0:
                out.append({"kind": "candidate", "text":
                            f"Adding {d['domain']} {lab}s to train would cover {c['delta_covered']} more production {lab}s."})
            if d["role"] == "train" and c["prod_lift"] is not None:
                if c["prod_lift"] >= 2 and c["rows"] and c["rows"] < np.median(train_rows[lab]):
                    out.append({"kind": "collect", "text":
                                f"{d['domain']} {lab}s look like production (lift {c['prod_lift']:.1f}) but there are "
                                f"only {c['rows']} rows: collect more like them."})
                if c["prod_lift"] < 0.3 and (c["delta_covered"] or 0) == 0:
                    out.append({"kind": "low", "text":
                                f"{d['domain']} {lab}s ({c['rows']} rows) are rarely near production (lift "
                                f"{c['prod_lift']:.2f}) and removing them uncovers no production sample: low production value."})
    separable = [f"{d['domain']} {lab}s ({c['auc_vs_prod']:.3f})" for d in doms for lab, c in d["cells"].items()
                 if d["role"] != "prod" and (c["auc_vs_prod"] or 0) >= 0.99]
    if separable:
        n_cells = sum(len(d["cells"]) for d in doms if d["role"] != "prod")
        out.append({"kind": "separable", "text":
                    f"{len(separable)} of {n_cells} non-production cells are linearly separable from production of the "
                    f"same label (AUC ≥ 0.99), i.e. a different domain to the model: " + ", ".join(separable) + "."})
    for o in data["overlaps"]:
        out.append({"kind": "leak", "text": f"{o['rows']} paths of {o['dropped_from']} also appear in {o['kept_in']} "
                                            f"(counted once, under {o['kept_in']})."})
    return out


def write_outputs(data: dict, res: dict, cfg: dict, args) -> None:
    doms = domain_rows(data, res, cfg)
    breakdown = prod_breakdown(data, res, args.min_support)
    clusters = cluster_rows(data, res, args)
    notes = findings(data, res, doms, breakdown, args)
    is_prod = data["role"] == "prod"
    kpis = {}
    for lab in (1, 0):
        sel = is_prod & (data["labels"] == lab)
        if sel.any():
            kpis[f"covered_{LABEL_NAMES[lab]}"] = r3(1 - res["uncovered"][sel].mean())
            kpis[f"prod_{LABEL_NAMES[lab]}"] = int(sel.sum())
    sub = res["sub"]
    meta_str = [" · ".join(f"{m}={data[f'meta_{m}'][i]}" for m in data["meta_names"] if data[f"meta_{m}"][i])
                for i in sub]
    points = {
        "x": [round(float(v), 4) for v in res["xy"][:, 0]], "y": [round(float(v), 4) for v in res["xy"][:, 1]],
        "cell": res["cell"][sub].tolist(), "cluster": res["cluster"][sub].tolist(),
        "coverage": [r3(min(v, 1.0)) if math.isfinite(v) else None for v in res["coverage"][sub]],
        "score": [r3(v) for v in data["scores"][sub]],
        "path": ["/".join(str(data["orig"][i]).split("/")[-4:]) for i in sub], "meta": meta_str,
    }
    cells = [{"domain": d, "label": LABEL_NAMES[l], "role": str(data["role"][res["cell"] == i][0]),
              "n": int((res["cell"] == i).sum())} for i, (d, l) in enumerate(res["cell_names"])]
    emb = data["embedding"]
    payload = {
        "meta": {"generated": time.strftime("%Y-%m-%d %H:%M"), "model": emb["model"], "image_size": emb["image_size"],
                 "resize_mode": emb["resize_mode"], "use_bbox_crop": emb["use_bbox_crop"], "k": args.k,
                 "dims": args.dims, "explained": r3(res["explained"]), "uncovered": args.uncovered,
                 "projection": res["projection"], "samples": int(len(data["labels"])), "config": str(args.config),
                 "sources": [{"name": s["name"], "role": s["role"], "csv": s["csv"]} for s in cfg["sources"]
                             if s["name"] in data["infos"]],
                 "has_scores": bool(np.isfinite(data["scores"]).any()), "has_thumbs": any(c["thumbs"] for c in clusters)},
        "kpis": kpis, "model": res["model"], "findings": notes, "domains": doms, "cells": cells,
        "affinity": {"order": res["order"], "log2_lift": [[r3(v) for v in row] for row in res["log_lift"]]},
        "prod_breakdown": breakdown, "clusters": clusters, "points": points,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in payload.items() if k != "points"}
    slim["clusters"] = [{k: v for k, v in c.items() if k != "thumbs"} for c in clusters]
    (args.out_dir / "summary.json").write_text(json.dumps(slim, indent=2))

    with (args.out_dir / "samples.csv").open("w", newline="") as f:
        w = csv.writer(f)
        cols = ["path", "source", "role", "domain", "label", "cluster", "coverage", "uncovered", "score", "error",
                "prod_nearest_domain", "x", "y"] + data["meta_names"]
        w.writerow(cols)
        pos = {int(i): j for j, i in enumerate(sub)}
        for i in range(len(data["labels"])):
            j = pos.get(i)
            w.writerow([data["orig"][i], data["source"][i], data["role"][i], data["domain"][i], data["labels"][i],
                        res["cluster"][i], r3(res["coverage"][i]), int(res["uncovered"][i]), r3(data["scores"][i]),
                        r3(res["error"][i]), res["prod_nearest"][i] or "",
                        r3(res["xy"][j, 0]) if j is not None else "", r3(res["xy"][j, 1]) if j is not None else ""]
                       + [data[f"meta_{m}"][i] for m in data["meta_names"]])

    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/"))
    (args.out_dir / "domains.html").write_text(html, encoding="utf-8")
    for note in notes:
        print(f"- {note['text']}")
    print(f"wrote {args.out_dir / 'domains.html'}, summary.json, samples.csv")


def run_report(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    data = load_embedded(cfg, args.out_dir / "emb", args.scores)
    res = analyse(data, cfg, args)
    write_outputs(data, res, cfg, args)


# -- CLI ------------------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    e = sub.add_parser("embed", help="Embed a sample of every registered source (needs the images and TensorFlow).")
    e.add_argument("--config", type=Path, required=True, help="Registry JSON, e.g. configs/domains.json.")
    e.add_argument("--out-dir", type=Path, required=True)
    e.add_argument("--checkpoint", type=Path, help="Trained .keras: embed with its pooled features.")
    e.add_argument("--imagenet", action="store_true", help="Embed with ImageNet EfficientNetB2 instead.")
    e.add_argument("--only", nargs="*", help="Source names to (re-)embed; default all that are out of date.")
    e.add_argument("--force", action="store_true", help="Re-embed even if up to date.")
    e.add_argument("--max-per-cell", type=int, default=1000, help="Rows sampled per (domain, label).")
    e.add_argument("--max-per-group", type=int, default=3, help="Frames per session/video.")
    e.add_argument("--thumb-size", type=int, default=0, help="Store whole-frame thumbnails, long side N px (0 = off).")
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--use-bbox-crop", action="store_true")
    e.add_argument("--margin", type=float, default=0.0)
    e.add_argument("--image-size", type=int, default=None)
    e.add_argument("--resize-mode", choices=("squash", "letterbox"), default=None)
    e.add_argument("--require-gpu", action="store_true")
    e.add_argument("--mixed-precision", action="store_true")

    r = sub.add_parser("report", help="Measure and write domains.html from the embeddings (no images needed).")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--out-dir", type=Path, required=True, help="The embed --out-dir; outputs go here too.")
    r.add_argument("--scores", type=Path, nargs="*", default=[], help="path,score CSVs to join by path.")
    r.add_argument("--k", type=int, default=15, help="Neighbours per sample.")
    r.add_argument("--dims", type=int, default=50, help="PCA dimensions before neighbours / clustering.")
    r.add_argument("--uncovered", type=float, default=0.25, help="Coverage below this = production-only region.")
    r.add_argument("--clusters", type=int, default=30)
    r.add_argument("--projection", choices=("auto", "umap", "tsne", "pca"), default="auto")
    r.add_argument("--map-max", type=int, default=15000, help="Points on the map (all production + a per-cell cap).")
    r.add_argument("--min-support", type=int, default=10, help="Smallest production group in the metadata breakdown.")
    r.add_argument("--thumbs-per-cluster", type=int, default=8)
    r.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_embed(args) if args.command == "embed" else run_report(args)


if __name__ == "__main__":
    main()
