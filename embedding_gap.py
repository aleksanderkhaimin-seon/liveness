#!/usr/bin/env python3
"""Measure how far apart two image domains sit in a trained model's embedding space.

Embeds each CSV (path,label[,bbox]) with the pooled backbone features (the input
of the final Dense layer), then reports, per class and overall:

  domain_auc     cross-validated AUC of a linear probe telling domain A from B
                 (0.5 = indistinguishable, 1.0 = trivially separable). The main number.
  centroid_dist  distance between domain means, in units of A's mean within-domain spread.
  frechet        Frechet distance between Gaussian fits (FID-style), on PCA-reduced features.
  knn_cross      fraction of each B point's 10 nearest neighbours that belong to A.

Run it on the same two CSVs for models trained with different --augment presets;
a preset that helps should lower domain_auc / frechet *and* prod EER. A lower gap
with an unchanged EER means the augmentation changed appearance, not what the model uses.

Example:
  python embedding_gap.py --checkpoint runs/x/best.keras \
      --a train.csv --b data/ProdTest-0.3_det_ik.csv --max-per-class 1500 \
      --out-dir runs/x/gap
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from input_geometry import find_report, resolve_geometry
from predict_checkpoint_csv import filter_missing_files, load_model, read_prediction_csv
from train_efficientnet_b2 import configure_runtime, make_dataset, set_input_geometry


def sample_rows(rows, labels, max_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    keep = []
    for cls in (0, 1):
        idx = np.flatnonzero(np.asarray(labels) == cls)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep.extend(idx.tolist())
    keep.sort()
    return keep


def embed(model: tf.keras.Model, csv_path: Path, args) -> tuple[np.ndarray, np.ndarray]:
    orig, resolved, labels, bboxes = read_prediction_csv(csv_path, use_bbox_crop=args.use_bbox_crop)
    orig, resolved, labels, bboxes = filter_missing_files(orig, resolved, labels, bboxes, "skip")
    keep = sample_rows(resolved, labels, args.max_per_class, args.seed)
    resolved = [resolved[i] for i in keep]
    bboxes = [bboxes[i] for i in keep]
    labels = np.asarray([labels[i] for i in keep])
    dataset = make_dataset(resolved, labels.tolist(), bboxes, args.batch_size, training=False,
                           use_bbox_crop=args.use_bbox_crop, margin=args.margin)
    # Feature input of the last Dense layer == pooled backbone output (Dropout is identity at inference).
    head = model.get_layer("live_score")
    feature_model = tf.keras.Model(model.input, head.input)
    feats = feature_model.predict(dataset, verbose=0).astype(np.float32)
    return feats, labels


def frechet(a: np.ndarray, b: np.ndarray, dims: int = 64) -> float:
    from scipy import linalg
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(dims, len(a) - 1, len(b) - 1)).fit(np.vstack([a, b]))
    a, b = pca.transform(a), pca.transform(b)
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root, _ = linalg.sqrtm(cov_a @ cov_b, disp=False)
    root = root.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * root))


def domain_auc(a: np.ndarray, b: np.ndarray, seed: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.vstack([a, b])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500, class_weight="balanced"))
    p = cross_val_predict(clf, x, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def knn_cross(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    from sklearn.neighbors import NearestNeighbors

    x = np.vstack([a, b])
    src = np.r_[np.zeros(len(a)), np.ones(len(b))]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
    idx = nn.kneighbors(b, return_distance=False)[:, 1:]  # drop self
    return float((src[idx] == 0).mean())


def gap_metrics(a, b, seed) -> dict:
    spread = float(np.sqrt(((a - a.mean(0)) ** 2).sum(1).mean()))
    return {
        "n_a": len(a), "n_b": len(b),
        "domain_auc": domain_auc(a, b, seed),
        "centroid_dist": float(np.linalg.norm(a.mean(0) - b.mean(0)) / spread),
        "frechet": frechet(a, b),
        "knn_cross": knn_cross(a, b),
        "knn_cross_chance": len(a) / (len(a) + len(b) - 1),
    }


def plot(fa, la, fb, lb, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    x = np.vstack([fa, fb])
    z = PCA(n_components=2).fit_transform(x)
    src = np.r_[np.zeros(len(fa)), np.ones(len(fb))]
    lab = np.r_[la, lb]
    fig, ax = plt.subplots(figsize=(7, 6))
    for s, name, marker in ((0, "A", "o"), (1, "B", "x")):
        for c, color in ((1, "tab:green"), (0, "tab:red")):
            m = (src == s) & (lab == c)
            ax.scatter(z[m, 0], z[m, 1], s=8, marker=marker, c=color, alpha=0.4,
                       label=f"{name} {'live' if c else 'attack'}")
    ax.legend(); ax.set_title("PCA of pooled embeddings")
    fig.savefig(path, dpi=130, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--a", type=Path, required=True, help="Reference domain CSV (e.g. train or in-distribution test).")
    ap.add_argument("--b", type=Path, required=True, help="Target domain CSV (e.g. production).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-per-class", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-bbox-crop", action="store_true")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--resize-mode", choices=("squash", "letterbox"), default=None)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--mixed-precision", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime(args.require_gpu, args.mixed_precision)
    report = find_report(args.checkpoint)
    geo = resolve_geometry(args.image_size, args.resize_mode, report_path=report)
    set_input_geometry(geo.image_size, geo.resize_mode)
    model = load_model(args.checkpoint)
    geo = resolve_geometry(args.image_size, args.resize_mode, model_input_size=int(model.input_shape[1]), report_path=report)
    set_input_geometry(geo.image_size, geo.resize_mode)

    fa, la = embed(model, args.a, args)
    fb, lb = embed(model, args.b, args)

    result = {"checkpoint": str(args.checkpoint), "a": str(args.a), "b": str(args.b),
              "overall": gap_metrics(fa, fb, args.seed)}
    for cls, name in ((1, "live"), (0, "attack")):
        if (la == cls).sum() > 20 and (lb == cls).sum() > 20:
            result[name] = gap_metrics(fa[la == cls], fb[lb == cls], args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "embeddings.npz", fa=fa, la=la, fb=fb, lb=lb)
    (args.out_dir / "gap.json").write_text(json.dumps(result, indent=2))
    plot(fa, la, fb, lb, args.out_dir / "pca.png")
    for key, val in result.items():
        if isinstance(val, dict):
            print(key, {k: round(v, 3) if isinstance(v, float) else v for k, v in val.items()})


if __name__ == "__main__":
    main()
