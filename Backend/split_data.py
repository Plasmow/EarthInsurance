"""Utilities to split datasets for occurrence (label) and magnitude (magnitude) tasks.

Creates:
  - train_occ.csv / test_occ.csv from combined_dataset.csv (stratified on 'label')
  - train_dmg.csv / test_dmg.csv from combined_data_set_tornado.csv (stratified on 'magnitude')

Usage examples (from repo root):
  py EarthInsurance/Backend/split_data.py split_occ \
	  --infile EarthInsurance/Backend/data/combined_dataset.csv \
	  --outdir EarthInsurance/Backend/data --val-size 0.25

  py EarthInsurance/Backend/split_data.py split_dmg \
	  --infile EarthInsurance/Backend/data/combined_data_set_tornado.csv \
	  --outdir EarthInsurance/Backend/data --val-size 0.25

Both commands print class distributions for train/validation.
"""

from __future__ import annotations
import argparse
import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def _stratified_split(df: pd.DataFrame, label_col: str, val_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
	if label_col not in df.columns:
		raise ValueError(f"Column '{label_col}' not found. Available: {list(df.columns)}")
	y = df[label_col]
	if y.nunique() < 2:
		# Fall back to non-stratified if only one class
		return train_test_split(df, test_size=val_size, random_state=seed, shuffle=True, stratify=None)
	return train_test_split(df, test_size=val_size, random_state=seed, shuffle=True, stratify=y)


def split_occ(infile: str, outdir: str, val_size: float, seed: int, train_name: str = "train_occ.csv", test_name: str = "test_occ.csv"):
	df = pd.read_csv(infile)
	train_df, val_df = _stratified_split(df, "label", val_size, seed)
	os.makedirs(outdir, exist_ok=True)
	train_path = os.path.join(outdir, train_name)
	val_path = os.path.join(outdir, test_name)
	train_df.to_csv(train_path, index=False)
	val_df.to_csv(val_path, index=False)
	print(f"[occ] Wrote {train_path} ({len(train_df)}) and {val_path} ({len(val_df)})")
	dist_tr = train_df['label'].value_counts(normalize=True).sort_index().to_dict()
	dist_va = val_df['label'].value_counts(normalize=True).sort_index().to_dict()
	print(f"[occ] Class distribution train: {dist_tr}")
	print(f"[occ] Class distribution val  : {dist_va}")


def split_dmg(infile: str, outdir: str, val_size: float, seed: int, train_name: str = "train_dmg.csv", test_name: str = "test_dmg.csv"):
	df = pd.read_csv(infile)
	if 'magnitude' not in df.columns:
		raise ValueError("Column 'magnitude' not found in damage dataset.")
	train_df, val_df = _stratified_split(df, "magnitude", val_size, seed)
	os.makedirs(outdir, exist_ok=True)
	train_path = os.path.join(outdir, train_name)
	val_path = os.path.join(outdir, test_name)
	train_df.to_csv(train_path, index=False)
	val_df.to_csv(val_path, index=False)
	print(f"[dmg] Wrote {train_path} ({len(train_df)}) and {val_path} ({len(val_df)})")
	dist_tr = train_df['magnitude'].value_counts(normalize=True).sort_index().to_dict()
	dist_va = val_df['magnitude'].value_counts(normalize=True).sort_index().to_dict()
	print(f"[dmg] Class distribution train: {dist_tr}")
	print(f"[dmg] Class distribution val  : {dist_va}")


def main():
	parser = argparse.ArgumentParser(description="Split datasets for occurrence and damage.")
	sub = parser.add_subparsers(dest="cmd")

	p_occ = sub.add_parser("split_occ", help="Split combined_dataset.csv for occurrence model")
	p_occ.add_argument("--infile", required=True)
	p_occ.add_argument("--outdir", required=True)
	p_occ.add_argument("--val-size", type=float, default=0.25)
	p_occ.add_argument("--seed", type=int, default=42)
	p_occ.add_argument("--train-name", default="train_occ.csv")
	p_occ.add_argument("--test-name", default="test_occ.csv")

	p_dmg = sub.add_parser("split_dmg", help="Split combined_data_set_tornado.csv for magnitude model")
	p_dmg.add_argument("--infile", required=True)
	p_dmg.add_argument("--outdir", required=True)
	p_dmg.add_argument("--val-size", type=float, default=0.25)
	p_dmg.add_argument("--seed", type=int, default=42)
	p_dmg.add_argument("--train-name", default="train_dmg.csv")
	p_dmg.add_argument("--test-name", default="test_dmg.csv")

	args = parser.parse_args()
	if args.cmd == "split_occ":
		split_occ(args.infile, args.outdir, args.val_size, args.seed, args.train_name, args.test_name)
	elif args.cmd == "split_dmg":
		split_dmg(args.infile, args.outdir, args.val_size, args.seed, args.train_name, args.test_name)
	else:
		parser.print_help()


if __name__ == "__main__":
	main()

