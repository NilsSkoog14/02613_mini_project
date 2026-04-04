from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 12 analysis")
    parser.add_argument("--csv", type=str, default="all_buildings_results.csv")
    parser.add_argument("--out-prefix", type=str, default="task12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    avg_mean_temp = df["mean_temp"].mean()
    avg_std_temp = df["std_temp"].mean()
    n_above_18 = int((df["pct_above_18"] >= 50.0).sum())
    n_below_15 = int((df["pct_below_15"] >= 50.0).sum())

    # Histogram for report question 12a.
    plt.figure(figsize=(8, 5))
    plt.hist(df["mean_temp"], bins=40, edgecolor="black")
    plt.xlabel("Mean interior temperature (degC)")
    plt.ylabel("Number of buildings")
    plt.title("Distribution of mean temperatures")
    plt.tight_layout()
    hist_path = f"{args.out_prefix}_mean_temp_hist.png"
    plt.savefig(hist_path, dpi=150)

    txt_path = Path(f"{args.out_prefix}_answers.txt")
    txt_path.write_text(
        "\n".join(
            [
                f"Buildings analyzed: {len(df)}",
                f"Average mean temperature: {avg_mean_temp:.6f}",
                f"Average std temperature: {avg_std_temp:.6f}",
                f"Buildings with >=50% area above 18C: {n_above_18}",
                f"Buildings with >=50% area below 15C: {n_below_15}",
                f"Histogram path: {hist_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(txt_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
