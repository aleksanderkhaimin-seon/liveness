#!/usr/bin/env python3
"""Per-zone and per-document EER for a patch-trained model.

Joins a test_predictions.csv (path,label,logit,score,prediction) produced by
predict_checkpoint_csv.py / predict_onnx_csv.py to the sampler's manifest.csv
on patch id, then reports:

  * per-patch EER and AUC, overall and by zone
  * per-document EER and AUC after aggregating patch scores with several rules
  * per-document EER by zone, to show where the signal lives

Per-patch EER is not the number that matters for a patch model: individual
tiles are legitimately ambiguous. Per-document EER after aggregation is the
number comparable with the full-frame pipeline.

Example:

    python aggregate_patch_eer.py runs/efficientnet_b2/exp_26/test_predictions.csv \\
        data/patches_test2/manifest.csv
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from eer import compute_eer

AGGREGATIONS = ("mean_score", "mean_logit", "median_score", "max_score", "top3_mean_score")


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score_pos > score_neg), ties count half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort().astype(np.float64) + 1.0
    # average ranks for ties
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_ranks = ranks[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            sorted_ranks[i : j + 1] = sorted_ranks[i : j + 1].mean()
        i = j + 1
    ranks[order] = sorted_ranks
    rank_sum_pos = ranks[: len(pos)].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    pos = scores[labels == 1]  # attacks, expected to score high
    neg = scores[labels == 0]  # bona fide
    if len(pos) == 0 or len(neg) == 0:
        return {"n_attack": len(pos), "n_bonafide": len(neg), "eer": float("nan"),
                "eer_threshold": float("nan"), "auc": float("nan")}
    eer, threshold = compute_eer(pos, neg)
    return {"n_attack": len(pos), "n_bonafide": len(neg), "eer": float(eer),
            "eer_threshold": float(threshold), "auc": auc_rank(pos, neg)}


def load_manifest(path: Path) -> dict[str, dict]:
    by_id = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            by_id[row["patch_id"]] = row
    return by_id


def load_predictions(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"path", "label", "score"}
        if not reader.fieldnames or required - set(reader.fieldnames):
            raise SystemExit(f"{path}: need columns {sorted(required)}, got {reader.fieldnames}")
        has_logit = "logit" in reader.fieldnames
        for row in reader:
            score = float(row["score"])
            if has_logit and row["logit"] not in ("", None):
                logit = float(row["logit"])
            else:
                clipped = min(max(score, 1e-7), 1 - 1e-7)
                logit = float(np.log(clipped / (1 - clipped)))
            rows.append({"patch_id": Path(row["path"]).stem, "label": int(row["label"]),
                         "score": score, "logit": logit})
    return rows


def aggregate(values: np.ndarray, logits: np.ndarray, rule: str) -> float:
    if rule == "mean_score":
        return float(values.mean())
    if rule == "mean_logit":
        return float(logits.mean())
    if rule == "median_score":
        return float(np.median(values))
    if rule == "max_score":
        return float(values.max())
    if rule == "top3_mean_score":
        return float(np.sort(values)[-3:].mean())
    raise ValueError(rule)


def fmt(m: dict) -> str:
    return (f"n_attack={m['n_attack']:>6}  n_bonafide={m['n_bonafide']:>6}  "
            f"EER={m['eer']:6.2f}%  AUC={m['auc']:.4f}  thr={m['eer_threshold']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("--min-patches", type=int, default=1,
                        help="Skip documents with fewer scored patches than this.")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="Write per-document aggregated scores here.")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest_csv)
    predictions = load_predictions(args.predictions_csv)

    joined = []
    missing = 0
    for row in predictions:
        meta = manifest.get(row["patch_id"])
        if meta is None:
            missing += 1
            continue
        if int(meta["label"]) != row["label"]:
            raise SystemExit(f"Label mismatch for {row['patch_id']}: manifest={meta['label']} predictions={row['label']}")
        joined.append({**row, "zone": meta["zone"], "source_path": meta["source_path"]})

    if not joined:
        raise SystemExit("No predictions matched the manifest. Are these the right two files?")
    if missing:
        print(f"WARNING: {missing} predictions had no manifest entry and were skipped", file=sys.stderr)

    scores = np.array([r["score"] for r in joined])
    logits = np.array([r["logit"] for r in joined])
    labels = np.array([r["label"] for r in joined])
    zones = np.array([r["zone"] for r in joined])

    print(f"Patches: {len(joined)}   documents: {len({r['source_path'] for r in joined})}\n")
    print("== per patch ==")
    print(f"  {'all':<10} {fmt(metrics(scores, labels))}")
    for zone in sorted(set(zones)):
        mask = zones == zone
        print(f"  {zone:<10} {fmt(metrics(scores[mask], labels[mask]))}")

    # group by document
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in joined:
        by_doc[row["source_path"]].append(row)

    print("\n== per document, all zones ==")
    doc_rows = []
    for rule in AGGREGATIONS:
        doc_scores, doc_labels = [], []
        for source, rows in by_doc.items():
            if len(rows) < args.min_patches:
                continue
            values = np.array([r["score"] for r in rows])
            lg = np.array([r["logit"] for r in rows])
            doc_scores.append(aggregate(values, lg, rule))
            doc_labels.append(rows[0]["label"])
        m = metrics(np.array(doc_scores), np.array(doc_labels))
        print(f"  {rule:<16} {fmt(m)}")
        if rule == "mean_logit":
            doc_rows = [(s, sc, lb) for s, sc, lb in zip(
                [k for k, v in by_doc.items() if len(v) >= args.min_patches], doc_scores, doc_labels)]

    print("\n== per document, by zone (mean_logit) ==")
    for zone in sorted(set(zones)):
        doc_scores, doc_labels = [], []
        for rows in by_doc.values():
            zone_rows = [r for r in rows if r["zone"] == zone]
            if len(zone_rows) < args.min_patches:
                continue
            doc_scores.append(float(np.mean([r["logit"] for r in zone_rows])))
            doc_labels.append(zone_rows[0]["label"])
        print(f"  {zone:<10} {fmt(metrics(np.array(doc_scores), np.array(doc_labels)))}")

    if args.out_csv:
        with args.out_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["source_path", "label", "mean_logit", "n_patches"])
            for source, score, label in doc_rows:
                writer.writerow([source, label, score, len(by_doc[source])])
        print(f"\nPer-document scores: {args.out_csv}")


if __name__ == "__main__":
    main()
