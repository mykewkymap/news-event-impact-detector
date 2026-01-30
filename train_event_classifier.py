#!/usr/bin/env python3
"""
Fine‑tune a FinBERT (or other Transformer) model to classify financial
news headlines into discrete event types.

This script expects a CSV dataset with columns ``headline`` and
``event_type`` where ``event_type`` is a categorical label such as
``earnings_beat``, ``guidance_cut``, ``litigation`` or ``fda_approval``.
It loads a pre‑trained model and tokenizer, tokenizes the data,
performs a train/validation split, fine‑tunes the model using the
Hugging Face ``Trainer`` API, and saves both the fine‑tuned model
weights and the label mapping to disk.

Example usage:

```
python train_event_classifier.py \
  --input_file data/event_classification_train.csv \
  --model_name ProsusAI/finbert \
  --output_dir models/event_classifier
```

``ProsusAI/finbert`` is a commonly used FinBERT model fine‑tuned on
financial sentiment【248260085656625†L68-L76】.  You may choose another model from
Hugging Face if desired.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from utils.text_utils import normalize_batch


class HeadlineDataset(Dataset):
    """Custom Dataset for financial news headlines."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_dataset(path: str) -> Tuple[List[str], List[str]]:
    """Load headlines and labels from a CSV file."""
    df = pd.read_csv(path)
    if "headline" not in df.columns or "event_type" not in df.columns:
        raise ValueError("Input file must contain 'headline' and 'event_type' columns")
    texts = normalize_batch(df["headline"].astype(str))
    labels = df["event_type"].astype(str).tolist()
    return texts, labels


def encode_labels(labels: List[str]) -> Tuple[List[int], Dict[str, int]]:
    """Encode string labels into integers and return the mapping."""
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = [label2id[label] for label in labels]
    return encoded, label2id


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an event classifier using FinBERT or another transformer model.")
    parser.add_argument("--input_file", type=str, required=True, help="CSV file with 'headline' and 'event_type' columns")
    parser.add_argument("--model_name", type=str, default="ProsusAI/finbert", help="Pre‑trained model name from Hugging Face hub")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine‑tuned model and label mapping")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for fine‑tuning")
    args = parser.parse_args()

    # Load and encode the dataset
    texts, string_labels = load_dataset(args.input_file)
    encoded_labels, label2id = encode_labels(string_labels)
    id2label = {v: k for k, v in label2id.items()}
    print(f"Loaded {len(texts)} samples across {len(label2id)} event classes")

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # Train/validation split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    train_dataset = HeadlineDataset(train_texts, train_labels, tokenizer)
    val_dataset = HeadlineDataset(val_texts, val_labels, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Define a simple compute_metric to report accuracy
    # Define compute_metrics without external libraries.  The Trainer API
    # expects a dict of metric name to value.  We compute accuracy
    # manually using NumPy.
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        accuracy = (predictions == labels).mean().item()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Training complete. Saving model and label mapping…")

    # Save the fine‑tuned model
    trainer.save_model(args.output_dir)
    # Save label mapping to JSON
    with open(f"{args.output_dir}/label_mapping.json", "w") as f:
        json.dump(label2id, f, indent=2)
    print(f"Saved model and label mapping to {args.output_dir}")


if __name__ == "__main__":
    main()