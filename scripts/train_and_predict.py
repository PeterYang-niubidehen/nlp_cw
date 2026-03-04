from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dpm_data import load_official_splits


def build_models() -> dict[str, Pipeline]:
    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    models = {
        "logreg": Pipeline(
            [
                ("tfidf", tfidf),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                        C=2.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "linearsvc": Pipeline(
            [
                ("tfidf", tfidf),
                ("clf", LinearSVC(class_weight="balanced", C=1.0, random_state=42)),
            ]
        ),
    }
    return models


def find_best_threshold(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_threshold = 0.0
    best_f1 = -1.0

    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def get_scores(model: Pipeline, x: pd.Series) -> np.ndarray:
    clf = model.named_steps["clf"]

    if hasattr(clf, "decision_function"):
        return model.decision_function(x)
    if hasattr(clf, "predict_proba"):
        probs = model.predict_proba(x)
        return probs[:, 1]

    return model.predict(x)


def save_predictions(path: Path, preds: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for p in preds:
            handle.write(f"{int(p)}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PCL classifier and generate predictions.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--bestmodel-dir", type=Path, default=Path("BestModel"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.bestmodel_dir.mkdir(parents=True, exist_ok=True)

    train_df, dev_df, test_df = load_official_splits(args.data_root)

    x_train = train_df["text"]
    y_train = train_df["binary_label"].to_numpy()

    x_dev = dev_df["text"]
    y_dev = dev_df["binary_label"].to_numpy()

    x_test = test_df["text"]

    model_pool = build_models()

    best_name = ""
    best_model = None
    best_threshold = 0.0
    best_dev_f1 = -1.0
    leaderboard = []

    for name, model in model_pool.items():
        model.fit(x_train, y_train)

        dev_scores = get_scores(model, x_dev)
        threshold, dev_f1 = find_best_threshold(dev_scores, y_dev)
        leaderboard.append((name, dev_f1, threshold))

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_threshold = threshold
            best_model = model
            best_name = name

    if best_model is None:
        raise RuntimeError("No model was successfully trained.")

    dev_scores = get_scores(best_model, x_dev)
    dev_preds = (dev_scores >= best_threshold).astype(int)

    test_scores = get_scores(best_model, x_test)
    test_preds = (test_scores >= best_threshold).astype(int)

    print("Candidate model dev F1 results:")
    for model_name, model_f1, threshold in sorted(leaderboard, key=lambda x: x[1], reverse=True):
        print(f"  {model_name:10s} F1={model_f1:.4f} threshold={threshold:.6f}")

    print("\nBest model:", best_name)
    print(f"Dev F1 (PCL=1): {f1_score(y_dev, dev_preds):.4f}")
    print("\nClassification report on dev set:")
    print(classification_report(y_dev, dev_preds, digits=4))

    save_predictions(args.outputs_dir / "dev.txt", dev_preds)
    save_predictions(args.outputs_dir / "test.txt", test_preds)

    joblib.dump(
        {
            "model_name": best_name,
            "pipeline": best_model,
            "threshold": best_threshold,
        },
        args.models_dir / "best_model.joblib",
    )

    dev_details = dev_df.copy()
    dev_details["y_true"] = y_dev
    dev_details["score"] = dev_scores
    dev_details["y_pred"] = dev_preds
    dev_details.to_csv(args.outputs_dir / "dev_predictions_detailed.csv", index=False)

    test_details = test_df.copy()
    test_details["score"] = test_scores
    test_details["y_pred"] = test_preds
    test_details.to_csv(args.outputs_dir / "test_predictions_detailed.csv", index=False)

    # Copy deliverables to BestModel folder expected by coursework.
    save_predictions(args.bestmodel_dir / "dev.txt", dev_preds)
    save_predictions(args.bestmodel_dir / "test.txt", test_preds)
    joblib.dump(
        {
            "model_name": best_name,
            "pipeline": best_model,
            "threshold": best_threshold,
        },
        args.bestmodel_dir / "best_model.joblib",
    )

    (args.bestmodel_dir / "MODEL_INFO.txt").write_text(
        "\n".join(
            [
                f"model={best_name}",
                f"dev_f1={f1_score(y_dev, dev_preds):.6f}",
                f"threshold={best_threshold:.8f}",
                "notes=trained on official train split and evaluated on official dev split",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
