#!/usr/bin/env python3
"""
Build clean mixed-domain train/val/test splits for CFN training/evaluation.

Outputs:
  - processed CSV splits for model training
  - val/test manifests with absolute video paths for full pipeline evaluation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.dataset_registry import get_dfdc_videos, get_fakeavceleb_videos


def _strata(df: pd.DataFrame) -> pd.Series:
    return df["dataset"].astype(str) + ":" + df["label"].astype(int).astype(str)


def _build_path_index(fakeav_root: Path, dfdc_root: Path) -> pd.DataFrame:
    rows = []
    for v in get_fakeavceleb_videos(str(fakeav_root)):
        rows.append(
            {
                "video_id": str(v["video_id"]),
                "dataset": "FakeAVCeleb",
                "label": int(v["label"]),
                "path": str(Path(v["path"]).resolve()),
            }
        )
    for v in get_dfdc_videos(str(dfdc_root)):
        rows.append(
            {
                "video_id": str(v["video_id"]),
                "dataset": "DFDC",
                "label": int(v["label"]),
                "path": str(Path(v["path"]).resolve()),
            }
        )
    out = pd.DataFrame(rows).drop_duplicates(subset=["video_id", "dataset", "label"])
    return out


def _save_manifest(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for _, r in df.iterrows():
            f.write(f"{r['path']}\t{int(r['label'])}\t{r['dataset']}\n")


def main():
    parser = argparse.ArgumentParser(description="Build mixed-domain train/val/test splits.")
    parser.add_argument("--processed-csv", default="data/processed/causal_multimodal_dataset.csv")
    parser.add_argument("--fakeav-root", default="data/raw/fakeavceleb")
    parser.add_argument("--dfdc-root", default="data/raw/dfdc")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-train-csv", default="data/processed/causal_multimodal_dataset_mixed_train.csv")
    parser.add_argument("--out-val-csv", default="data/processed/causal_multimodal_dataset_mixed_val.csv")
    parser.add_argument("--out-test-csv", default="data/processed/causal_multimodal_dataset_mixed_test.csv")
    parser.add_argument("--out-val-manifest", default="eval_manifest_mixed_val.tsv")
    parser.add_argument("--out-test-manifest", default="eval_manifest_mixed_test.tsv")
    args = parser.parse_args()

    proc_path = Path(args.processed_csv)
    proc = pd.read_csv(proc_path)
    required = {"video_id", "dataset", "label"}
    missing = required - set(proc.columns)
    if missing:
        raise RuntimeError(f"Processed CSV missing columns: {sorted(missing)}")

    key_cols = ["video_id", "dataset", "label"]
    keys = proc[key_cols].drop_duplicates().reset_index(drop=True)
    keys["label"] = keys["label"].astype(int)

    strat = _strata(keys)
    train_val, test = train_test_split(
        keys,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=strat,
    )
    rel_val_size = args.val_size / (1.0 - args.test_size)
    train, val = train_test_split(
        train_val,
        test_size=rel_val_size,
        random_state=args.seed,
        stratify=_strata(train_val),
    )

    def _filter_split(split_df):
        split_keys = set(map(tuple, split_df[key_cols].itertuples(index=False, name=None)))
        mask = proc[key_cols].apply(tuple, axis=1).isin(split_keys)
        return proc.loc[mask].copy()

    train_proc = _filter_split(train)
    val_proc = _filter_split(val)
    test_proc = _filter_split(test)

    out_train = Path(args.out_train_csv)
    out_val = Path(args.out_val_csv)
    out_test = Path(args.out_test_csv)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    train_proc.to_csv(out_train, index=False)
    val_proc.to_csv(out_val, index=False)
    test_proc.to_csv(out_test, index=False)

    path_idx = _build_path_index(Path(args.fakeav_root), Path(args.dfdc_root))
    val_manifest_df = val.merge(path_idx, on=key_cols, how="left")
    test_manifest_df = test.merge(path_idx, on=key_cols, how="left")

    val_missing = int(val_manifest_df["path"].isna().sum())
    test_missing = int(test_manifest_df["path"].isna().sum())
    val_manifest_df = val_manifest_df.dropna(subset=["path"])
    test_manifest_df = test_manifest_df.dropna(subset=["path"])

    _save_manifest(val_manifest_df, Path(args.out_val_manifest))
    _save_manifest(test_manifest_df, Path(args.out_test_manifest))

    print("Split sizes (unique videos):")
    print("  train:", len(train), "val:", len(val), "test:", len(test))
    print("Processed rows:")
    print("  train:", len(train_proc), "val:", len(val_proc), "test:", len(test_proc))
    print("Manifest rows:")
    print("  val:", len(val_manifest_df), "test:", len(test_manifest_df))
    print("Missing paths dropped:")
    print("  val:", val_missing, "test:", test_missing)
    print("Outputs:")
    print(" ", out_train)
    print(" ", out_val)
    print(" ", out_test)
    print(" ", Path(args.out_val_manifest))
    print(" ", Path(args.out_test_manifest))


if __name__ == "__main__":
    main()
