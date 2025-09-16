"""Utility script to validate DataTransformer round-trip on `train-v4.csv`."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from Data_transformer_fix_gmm import DataTransformer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("train-v4-filtered.csv"),
        help="Path to the CSV file containing the training data.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Optional number of rows to sample for the check (0 means use full dataset).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used when sampling rows.",
    )
    return parser.parse_args()


def build_transformer():
    discrete_columns = ["spg"] + [f"wp{i}" for i in range(8)]
    grouped_discrete_columns = [discrete_columns[1:]]

    xyz_columns = [
        f"{axis}{idx}" for idx in range(8) for axis in ("x", "y", "z")
    ]
    grouped_continuous_columns = [
        #["a", "b", "c"],
        #["alpha", "beta", "gamma"],
        xyz_columns,
    ]

    return DataTransformer(max_clusters=10, weight_threshold=0.00001,
        grouped_continuous_columns=grouped_continuous_columns,
        grouped_discrete_columns=grouped_discrete_columns,
    ), discrete_columns, grouped_continuous_columns


def summarize_differences(original_df, recovered_df, discrete_columns):
    mismatched_discrete = {}
    for column in discrete_columns:
        mismatch_mask = original_df[column] != recovered_df[column]
        mismatched_discrete[column] = int(mismatch_mask.sum())

    continuous_columns = [
        column for column in original_df.columns if column not in discrete_columns
    ]
    continuous_diff = (original_df[continuous_columns] - recovered_df[continuous_columns]).abs()
    for column in continuous_columns:
        nonzero_count = (continuous_diff[column] > 1e-8).sum()
        print(f"Continuous column '{column}': nonzero diff count = {nonzero_count}")

    if continuous_diff.empty:
        max_abs_diff = 0.0
        mean_abs_diff = 0.0
        per_column_max = {}
    else:
        max_abs_diff = continuous_diff.max().max()
        mean_abs_diff = continuous_diff.mean().mean()
        per_column_max = continuous_diff.max().to_dict()

    return mismatched_discrete, float(max_abs_diff), float(mean_abs_diff), per_column_max


def main():
    args = parse_args()
    data_path = args.data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file at {data_path}")

    df = pd.read_csv(data_path).iloc[:,:-2] 
    if args.sample:
        df = df.sample(args.sample, random_state=args.random_state)

    df = df.reset_index(drop=True)

    transformer, discrete_columns, grouped_continuous_columns = build_transformer()

    transformer.fit(
        df,
        discrete_columns=discrete_columns,
        grouped_continuous_columns=grouped_continuous_columns,
        grouped_discrete_columns=[discrete_columns[1:]],
    )
    print('output_info_list:', transformer.output_info_list)

    transformed = transformer.transform(df)
    recovered = transformer.inverse_transform(transformed)

    (
        mismatched_discrete,
        max_abs_diff,
        mean_abs_diff,
        per_column_max_diff,
    ) = summarize_differences(df, recovered, discrete_columns)

    print("DataTransformer round-trip validation")
    print(f"Rows checked: {len(df)}")
    print(f"Transformed shape: {transformed.shape}")
    print("Discrete mismatches per column:")
    for column, count in mismatched_discrete.items():
        print(f"  {column}: {count}")


    print(f"Continuous max abs diff: {max_abs_diff:.6f}")
    print(f"Continuous mean abs diff: {mean_abs_diff:.6f}")

    total_mismatches = sum(mismatched_discrete.values())
    if total_mismatches == 0 and np.isclose(max_abs_diff, 0.0):
        print("SUCCESS: Transform and inverse transform preserve all values.")
    else:
        if per_column_max_diff:
            top_columns = sorted(
                per_column_max_diff.items(), key=lambda item: item[1], reverse=True
            )
            print("Top continuous column max abs differences:")
            for column, diff in top_columns:
                print(f"  {column}: {diff:.6f}")

        print("WARNING: Differences detected during round-trip. Inspect the details above.")


if __name__ == "__main__":
    main()
