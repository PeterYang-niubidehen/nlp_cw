from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate error analysis for dev predictions.")
    parser.add_argument(
        "--dev-details",
        type=Path,
        default=Path("outputs/dev_predictions_detailed.csv"),
        help="CSV produced by scripts/train_and_predict.py",
    )
    parser.add_argument(
        "--score-col",
        type=str,
        default="auto",
        help="Column used as confidence score. Use 'auto' to prefer score, then prob_pos.",
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    if not args.dev_details.exists():
        raise FileNotFoundError(
            f"Missing {args.dev_details}. Run scripts/train_and_predict.py first."
        )

    df = pd.read_csv(args.dev_details)
    required = {"y_true", "y_pred", "text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dev details file: {missing}")

    if args.score_col == "auto":
        if "score" in df.columns:
            score_col = "score"
        elif "prob_pos" in df.columns:
            score_col = "prob_pos"
        else:
            raise ValueError("No score column found. Expected either 'score' or 'prob_pos'.")
    else:
        score_col = args.score_col
        if score_col not in df.columns:
            raise ValueError(f"Requested --score-col '{score_col}' not found in CSV columns.")

    y_true = df["y_true"].astype(int)
    y_pred = df["y_pred"].astype(int)

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No PCL (0)", "PCL (1)"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Dev Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.figures_dir / "confusion_matrix_dev.png", dpi=200)
    plt.close()

    fp = df[(df["y_true"] == 0) & (df["y_pred"] == 1)].copy()
    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].copy()

    # Rank high-confidence mistakes first.
    fp["mistake_strength"] = fp[score_col]
    fn["mistake_strength"] = -fn[score_col]

    fp = fp.sort_values("mistake_strength", ascending=False)
    fn = fn.sort_values("mistake_strength", ascending=False)

    fp.head(50).to_csv(args.reports_dir / "false_positives_top50.csv", index=False)
    fn.head(50).to_csv(args.reports_dir / "false_negatives_top50.csv", index=False)

    # Score distribution by class/prediction for additional local evaluation.
    score_plot_df = df.copy()
    score_plot_df["group"] = (
        "true=" + score_plot_df["y_true"].astype(str) + ",pred=" + score_plot_df["y_pred"].astype(str)
    )
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=score_plot_df, x=score_col, hue="group", fill=True, common_norm=False, alpha=0.3)
    plt.title("Decision Score Distribution by Outcome Group")
    plt.tight_layout()
    plt.savefig(args.figures_dir / "score_distribution_by_outcome.png", dpi=200)
    plt.close()

    report_lines = [
        "# Local Evaluation Report",
        "",
        f"- F1 (PCL=1): {f1:.4f}",
        f"- Precision (PCL=1): {precision:.4f}",
        f"- Recall (PCL=1): {recall:.4f}",
        f"- False Positives: {len(fp)}",
        f"- False Negatives: {len(fn)}",
        f"- Score Column: {score_col}",
        "",
        "## Error Analysis",
        "- Inspect false_positives_top50.csv for cases where the model over-detects PCL.",
        "- Inspect false_negatives_top50.csv for missed PCL cases.",
        "",
        "## Additional Local Evaluation",
        "- confusion_matrix_dev.png shows class-level trade-offs.",
        "- score_distribution_by_outcome.png shows score separability and overlap.",
        "",
        "## Suggested Discussion Points",
        "- Are false negatives concentrated in subtle/implicit patronising language?",
        "- Are false positives caused by keywords that are emotionally loaded but not condescending?",
        "- What preprocessing or training changes can reduce these specific errors?",
    ]

    (args.reports_dir / "LOCAL_EVALUATION.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("Error analysis complete.")
    print(f"Report: {args.reports_dir / 'LOCAL_EVALUATION.md'}")


if __name__ == "__main__":
    main()
