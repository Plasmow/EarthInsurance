import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Split a single CSV (trainingandtest.csv) into train/validation (75/25) stratified on 'label'."
    )
    parser.add_argument("--infile", default="data/trainingandtest.csv", help="Input CSV (default: trainingandtest.csv)")
    parser.add_argument("--outdir", default="data", help="Output directory (default: data)")
    parser.add_argument("--train-name", default="train.csv", help="Train output filename")
    parser.add_argument("--val-name", default="test.csv", help="Validation output filename")
    parser.add_argument("--val-size", type=float, default=0.25, help="Validation ratio (default: 0.25)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.infile)
    if "label" not in df.columns:
        raise ValueError("Column 'label' not found in input CSV.")

    # Check if stratification is possible (need at least 2 classes)
    nunique = df["label"].nunique()
    stratify_series = df["label"] if nunique > 1 else None

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
        stratify=stratify_series,
    )

    train_path = os.path.join(args.outdir, args.train_name)
    val_path = os.path.join(args.outdir, args.val_name)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Train: {train_path} ({len(train_df)})")
    print(f"Validation: {val_path} ({len(val_df)})")

    if stratify_series is not None:
        print("Class distribution (train):", train_df["label"].value_counts(normalize=True).to_dict())
        print("Class distribution (val):", val_df["label"].value_counts(normalize=True).to_dict())
    else:
        print("Stratification skipped (only one class present).")


def main2():
    parser = argparse.ArgumentParser(
        description="Split a single CSV (trainingandtest2.csv) into train/validation (75/25) stratified on 'label'."
    )
    parser.add_argument("--infile", default="data/trainingandtest2.csv", help="Input CSV (default: trainingandtest.csv)")
    parser.add_argument("--outdir", default="data", help="Output directory (default: data)")
    parser.add_argument("--train-name", default="train_dmg.csv", help="Train output filename")
    parser.add_argument("--val-name", default="test_dmg.csv", help="Validation output filename")
    parser.add_argument("--val-size", type=float, default=0.25, help="Validation ratio (default: 0.25)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.infile)
    if "label" not in df.columns:
        raise ValueError("Column 'label' not found in input CSV.")

    # Check if stratification is possible (need at least 2 classes)
    nunique = df["label"].nunique()
    stratify_series = df["label"] if nunique > 1 else None

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
        stratify=stratify_series,
    )

    train_path = os.path.join(args.outdir, args.train_name)
    val_path = os.path.join(args.outdir, args.val_name)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Train: {train_path} ({len(train_df)})")
    print(f"Validation: {val_path} ({len(val_df)})")

    if stratify_series is not None:
        print("Class distribution (train):", train_df["label"].value_counts(normalize=True).to_dict())
        print("Class distribution (val):", val_df["label"].value_counts(normalize=True).to_dict())
    else:
        print("Stratification skipped (only one class present).")


if __name__ == "__main__":
    main()