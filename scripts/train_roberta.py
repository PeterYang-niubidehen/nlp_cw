from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dpm_data import load_official_splits


@dataclass
class EncodedBatch:
    input_ids: list[list[int]]
    attention_mask: list[list[int]]


class TextClassificationDataset(Dataset):
    def __init__(self, encodings: EncodedBatch, labels: np.ndarray | None = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings.input_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "input_ids": torch.tensor(self.encodings.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings.attention_mask[idx], dtype=torch.long),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: torch.Tensor | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is None:
            loss_fn = CrossEntropyLoss()
        else:
            loss_fn = CrossEntropyLoss(weight=self.class_weights.to(logits.device))

        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        if return_outputs:
            return loss, outputs
        return loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enrich_text(df: pd.DataFrame, use_meta_tokens: bool) -> pd.Series:
    text = df["text"].fillna("").astype(str)
    if not use_meta_tokens:
        return text
    return (
        "kw_"
        + df["keyword"].fillna("unk").astype(str)
        + " c_"
        + df["country"].fillna("unk").astype(str)
        + " "
        + text
    )


def encode_texts(tokenizer: Any, texts: list[str], max_length: int) -> EncodedBatch:
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    return EncodedBatch(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def tune_threshold(y_true: np.ndarray, pos_scores: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.01, 0.99, 199):
        preds = (pos_scores >= threshold).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def compute_metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = softmax(logits)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "f1_pos": f1_score(labels, preds),
        "precision_pos": precision_score(labels, preds, zero_division=0),
        "recall_pos": recall_score(labels, preds, zero_division=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model for PCL classification.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/roberta"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/roberta"))
    parser.add_argument("--bestmodel-dir", type=Path, default=Path("BestModel"))
    parser.add_argument("--model-name", type=str, default="distilroberta-base")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-meta-tokens", action="store_true")
    parser.add_argument("--save-bestmodel", action="store_true", help="Overwrite BestModel artifacts with this run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.bestmodel_dir.mkdir(parents=True, exist_ok=True)

    train_df, dev_df, test_df = load_official_splits(args.data_root)

    train_texts = enrich_text(train_df, args.use_meta_tokens).tolist()
    dev_texts = enrich_text(dev_df, args.use_meta_tokens).tolist()
    test_texts = enrich_text(test_df, args.use_meta_tokens).tolist()

    y_train = train_df["binary_label"].astype(int).to_numpy()
    y_dev = dev_df["binary_label"].astype(int).to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_enc = encode_texts(tokenizer, train_texts, args.max_length)
    dev_enc = encode_texts(tokenizer, dev_texts, args.max_length)
    test_enc = encode_texts(tokenizer, test_texts, args.max_length)

    train_dataset = TextClassificationDataset(train_enc, y_train)
    dev_dataset = TextClassificationDataset(dev_enc, y_dev)
    test_dataset = TextClassificationDataset(test_enc, None)

    steps_per_epoch = int(np.ceil(len(train_dataset) / args.train_batch_size))
    total_train_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = max(1, int(total_train_steps * args.warmup_ratio))

    class_counts = np.bincount(y_train, minlength=2)
    class_weights_np = len(y_train) / (2.0 * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(args.models_dir),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_pos",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        use_cpu=not torch.cuda.is_available(),
        do_train=True,
        do_eval=True,
        fp16=False,
        bf16=False,
    )

    def hf_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, labels = eval_pred
        return compute_metrics_from_logits(logits, labels)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=hf_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    dev_logits = trainer.predict(dev_dataset).predictions
    dev_probs = softmax(dev_logits)[:, 1]
    threshold, tuned_f1 = tune_threshold(y_dev, dev_probs)
    dev_preds = (dev_probs >= threshold).astype(int)

    print(f"Best threshold on dev: {threshold:.4f}")
    print(f"Dev F1 (PCL=1): {f1_score(y_dev, dev_preds):.4f}")
    print(classification_report(y_dev, dev_preds, digits=4))

    test_logits = trainer.predict(test_dataset).predictions
    test_probs = softmax(test_logits)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)

    def write_preds(path: Path, preds: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for p in preds:
                handle.write(f"{int(p)}\n")

    write_preds(args.output_dir / "dev.txt", dev_preds)
    write_preds(args.output_dir / "test.txt", test_preds)

    dev_details = dev_df.copy()
    dev_details["y_true"] = y_dev
    dev_details["prob_pos"] = dev_probs
    dev_details["y_pred"] = dev_preds
    dev_details.to_csv(args.output_dir / "dev_predictions_detailed.csv", index=False)

    test_details = test_df.copy()
    test_details["prob_pos"] = test_probs
    test_details["y_pred"] = test_preds
    test_details.to_csv(args.output_dir / "test_predictions_detailed.csv", index=False)

    metadata = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "seed": args.seed,
        "use_meta_tokens": args.use_meta_tokens,
        "class_weights": class_weights_np.tolist(),
        "dev_f1_best_threshold": tuned_f1,
        "best_threshold": threshold,
        "dev_positive_rate_pred": float(np.mean(dev_preds)),
        "test_positive_rate_pred": float(np.mean(test_preds)),
    }

    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    trainer.save_model(str(args.models_dir / "checkpoint_best"))
    tokenizer.save_pretrained(str(args.models_dir / "checkpoint_best"))
    joblib.dump(metadata, args.models_dir / "run_metadata.joblib")

    if args.save_bestmodel:
        write_preds(args.bestmodel_dir / "dev.txt", dev_preds)
        write_preds(args.bestmodel_dir / "test.txt", test_preds)
        best_model_path = (args.models_dir / "checkpoint_best").resolve()
        joblib.dump(metadata, args.bestmodel_dir / "best_model.joblib")
        (args.bestmodel_dir / "TRANSFORMER_MODEL_PATH.txt").write_text(
            str(best_model_path) + "\n",
            encoding="utf-8",
        )
        (args.bestmodel_dir / "MODEL_INFO.txt").write_text(
            "\n".join(
                [
                    f"model={args.model_name}",
                    f"dev_f1={f1_score(y_dev, dev_preds):.6f}",
                    f"threshold={threshold:.8f}",
                    f"max_length={args.max_length}",
                    f"epochs={args.epochs}",
                    f"train_batch_size={args.train_batch_size}",
                    f"use_meta_tokens={args.use_meta_tokens}",
                    f"model_checkpoint_path={best_model_path}",
                    "notes=transformer fine-tuning with class-weighted loss and threshold tuning",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
