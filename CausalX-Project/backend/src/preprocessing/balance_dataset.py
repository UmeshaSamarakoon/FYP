import argparse
import pandas as pd


def balance_dataset(input_csv, output_csv, label_col="label", seed=42):
    df = pd.read_csv(input_csv)

    if label_col not in df.columns:
        raise ValueError(f"Missing '{label_col}' column in {input_csv}")

    counts = df[label_col].value_counts()
    if counts.empty or len(counts) < 2:
        raise ValueError("Dataset must contain at least two classes to balance.")

    min_count = counts.min()
    balanced_parts = []

    for label_value, group in df.groupby(label_col):
        balanced_parts.append(group.sample(min_count, random_state=seed))

    balanced = pd.concat(balanced_parts).sample(frac=1, random_state=seed)
    balanced.to_csv(output_csv, index=False)

    print("Saved balanced dataset:", output_csv)
    print("Class counts:")
    print(balanced[label_col].value_counts())


def main():
    parser = argparse.ArgumentParser(description="Balance dataset by downsampling to the smallest class.")
    parser.add_argument("--input-csv", default="data/processed/causal_multimodal_dataset.csv")
    parser.add_argument("--output-csv", default="data/processed/causal_multimodal_dataset_balanced.csv")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    balance_dataset(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        label_col=args.label_col,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
