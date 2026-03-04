from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dpm_data import load_official_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA plots/tables for PCL dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    train_df, dev_df, _ = load_official_splits(args.data_root)

    train_df = train_df.copy()
    dev_df = dev_df.copy()

    train_df["split"] = "train"
    dev_df["split"] = "dev"
    full = pd.concat([train_df, dev_df], ignore_index=True)

    full["token_count"] = full["text"].astype(str).str.split().apply(len)

    # EDA 1: Class distribution table + plot.
    class_dist = (
        full.groupby(["split", "binary_label"]).size().rename("count").reset_index()
    )
    class_pivot = class_dist.pivot(index="split", columns="binary_label", values="count").fillna(0)
    class_pivot.columns = ["label_0_no_pcl", "label_1_pcl"]
    class_pivot["pcl_ratio"] = class_pivot["label_1_pcl"] / (
        class_pivot["label_0_no_pcl"] + class_pivot["label_1_pcl"]
    )
    class_pivot.to_csv(args.reports_dir / "eda_class_distribution.csv")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=class_dist, x="split", y="count", hue="binary_label")
    plt.title("Class Distribution by Split")
    plt.xlabel("Split")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(args.figures_dir / "eda_class_distribution.png", dpi=200)
    plt.close()

    # EDA 2: Token-length distribution by class.
    plt.figure(figsize=(9, 5))
    sns.kdeplot(
        data=full,
        x="token_count",
        hue="binary_label",
        fill=True,
        common_norm=False,
        alpha=0.35,
    )
    plt.title("Token Count Distribution by Class")
    plt.xlabel("Token count")
    plt.ylabel("Density")
    plt.xlim(0, full["token_count"].quantile(0.99))
    plt.tight_layout()
    plt.savefig(args.figures_dir / "eda_token_count_distribution.png", dpi=200)
    plt.close()

    token_stats = (
        full.groupby("binary_label")["token_count"]
        .agg(["count", "mean", "median", "min", "max"]) 
        .reset_index()
    )
    token_stats.to_csv(args.reports_dir / "eda_token_count_stats.csv", index=False)

    # Additional lexical evidence for optional use in report.
    keyword_dist = (
        full.groupby(["binary_label", "text"]) 
        .size() 
        .sort_values(ascending=False) 
        .head(20)
        .reset_index(name="frequency")
    )
    keyword_dist.to_csv(args.reports_dir / "eda_top_repeated_texts.csv", index=False)

    (args.reports_dir / "EDA_NOTES.md").write_text(
        "\n".join(
            [
                "# EDA Notes",
                "",
                "## Technique 1: Class Distribution",
                "Evidence: eda_class_distribution.csv + eda_class_distribution.png",
                "Insight: PCL is minority class, so class imbalance handling is required.",
                "Impact: Use class_weight='balanced' and optimize threshold for F1(positive class).",
                "",
                "## Technique 2: Text Length Distribution",
                "Evidence: eda_token_count_distribution.png + eda_token_count_stats.csv",
                "Insight: Label-specific length differences can affect model behavior.",
                "Impact: Keep tokenization robust and include n-grams to capture short and long cues.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("EDA complete.")
    print(f"Figures written to: {args.figures_dir}")
    print(f"Tables/notes written to: {args.reports_dir}")


if __name__ == "__main__":
    main()
