import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def downsample_majority(df: pd.DataFrame, label_col: str, seed: int) -> pd.DataFrame:
    """
    Downsample majority classes to match the minority count.
    """
    groups = df.groupby(label_col)
    min_count = groups.size().min()
    balanced_parts = []
    rng = np.random.default_rng(seed)
    for _, g in groups:
        if len(g) > min_count:
            balanced_parts.append(g.sample(min_count, random_state=int(rng.integers(0, 1e9))))
        else:
            balanced_parts.append(g)
    return pd.concat(balanced_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Downsample dataset to a balanced class distribution.")
    parser.add_argument("--input-csv", required=True, help="Path to input CSV with a 'label' column.")
    parser.add_argument("--output-csv", required=True, help="Path to write the balanced CSV.")
    parser.add_argument("--label-col", default="label", help="Label column name (default: label).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    inp = Path(args.input_csv)
    out = Path(args.output_csv)

    df = pd.read_csv(inp)
    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in {inp}")

    balanced = downsample_majority(df, args.label_col, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(out, index=False)

    # Report class counts
    counts = balanced[args.label_col].value_counts().to_dict()
    print(f"Balanced dataset saved to {out} | counts: {counts} | rows: {len(balanced)}")


if __name__ == "__main__":
    main()
