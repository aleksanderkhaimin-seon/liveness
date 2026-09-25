#!/usr/bin/env python3
"""Drop datasets from a liveness train CSV and balance classes."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _stratified_sample(df: pd.DataFrame, n: int, seed: int, replace: bool) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0]
    if "dataset" not in df.columns or df["dataset"].nunique() <= 1:
        return df.sample(n=n, replace=replace, random_state=seed)

    parts = []
    allocated = 0
    groups = list(df.groupby("dataset"))
    for i, (_, group) in enumerate(groups):
        if i == len(groups) - 1:
            take = n - allocated
        else:
            take = round(len(group) / len(df) * n)
        if not replace:
            take = min(take, len(group))
        take = max(0, take)
        if take <= 0:
            continue
        parts.append(group.sample(n=take, replace=replace, random_state=seed))
        allocated += take
    sampled = pd.concat(parts)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    elif len(sampled) < n:
        sampled = pd.concat(
            [sampled, df.sample(n=n - len(sampled), replace=True, random_state=seed)]
        )
    return sampled


def undersample_majority(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = df["label"].value_counts()
    minority_label = int(counts.idxmin())
    majority_label = int(counts.idxmax())
    n_keep = int(counts[minority_label])
    minority = df[df["label"] == minority_label]
    majority = _stratified_sample(df[df["label"] == majority_label], n_keep, seed, replace=False)
    out = pd.concat([minority, majority], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def oversample_minority(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = df["label"].value_counts()
    minority_label = int(counts.idxmin())
    majority_label = int(counts.idxmax())
    n_keep = int(counts[majority_label])
    majority = df[df["label"] == majority_label]
    minority = _stratified_sample(df[df["label"] == minority_label], n_keep, seed, replace=True)
    out = pd.concat([majority, minority], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/train_Seon-DocX-Pint_Unidata_DocXPand_exp__checked.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/train_Seon-Pint_Unidata_no-docxpand_balanced.csv"),
    )
    parser.add_argument("--exclude-dataset", action="append", default=None)
    parser.add_argument(
        "--method",
        choices=("undersample", "oversample"),
        default="undersample",
        help="undersample majority (drops live) or oversample minority (keeps all live).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv, dtype={"bbox": "string"})
    print(f"input: {len(df):,} rows")
    print(df.groupby(["dataset", "label"]).size().unstack(fill_value=0).to_string())

    exclude = args.exclude_dataset or []
    if exclude:
        excluded = df["dataset"].isin(exclude)
        print(f"\nexclude {exclude}: {int(excluded.sum()):,} rows")
        df = df.loc[~excluded].copy()
        print(f"after exclude: {len(df):,}")
    print("labels:", df["label"].value_counts().sort_index().to_dict())

    if args.method == "oversample":
        balanced = oversample_minority(df, args.seed)
    else:
        balanced = undersample_majority(df, args.seed)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(args.output_csv, index=False)

    print(f"\nwrote {args.output_csv} ({len(balanced):,} rows)")
    print("labels:", balanced["label"].value_counts().sort_index().to_dict())
    print(balanced.groupby(["dataset", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
