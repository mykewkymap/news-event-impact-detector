#!/usr/bin/env python3
"""
Inference script for the News Event Impact Detector.

Given a list of news events (with timestamps, tickers and headlines),
this script predicts the most probable event type using a
fine‑tuned transformer model and then estimates the expected stock
return impact using a trained regression model.  It outputs a table
with the predicted event type, expected return and a qualitative
reason based on historical averages.

Example usage:

```
python predict.py \
  --news_file data/new_events.csv \
  --event_model models/event_classifier \
  --impact_model models/impact_model.pkl
```

The news CSV should contain columns ``timestamp``, ``ticker`` and
``headline``.  Optionally, you can provide ``--top_k`` to return the
top k probable event types and their probabilities for each
headline.  By default only the most likely event and its impact are
returned.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.text_utils import normalize_batch

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

import joblib


def load_event_classifier(model_dir: str):
    """Load the fine‑tuned event classifier and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    id2label = model.config.id2label
    return tokenizer, model, id2label


def load_impact_model(model_path: str):
    """Load the impact regression model and its metadata."""
    # Determine model type from file extension or metadata
    if model_path.endswith(".pkl"):
        # joblib saved dictionary
        data = joblib.load(model_path)
        model = data["model"]
        feature_columns = data["feature_columns"]
        model_type = data["model_type"]
    else:
        # Assume LightGBM
        model = lgb.Booster(model_file=model_path)
        meta_file = model_path + ".meta.json"
        with open(meta_file) as f:
            meta = json.load(f)
        feature_columns = meta["feature_columns"]
        model_type = meta["model_type"]
    # Load event effects
    effects_file = model_path + ".effects.json"
    if os.path.exists(effects_file):
        with open(effects_file) as f:
            event_effects = json.load(f)
    else:
        event_effects = {}
    return model, feature_columns, model_type, event_effects


def predict_events(tokenizer, model, texts: List[str], top_k: int = 1) -> Tuple[List[str], List[List[Tuple[str, float]]]]:
    """Predict event type(s) for a list of headlines.

    Returns both the top label and optionally the top_k label probabilities.
    """
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits.detach().cpu().numpy()
    probs = softmax(logits)
    top_labels = []
    top_k_probs = []
    id2label = model.config.id2label
    for i in range(len(texts)):
        prob_i = probs[i]
        sorted_idx = np.argsort(prob_i)[::-1]
        label = id2label[str(sorted_idx[0]) if isinstance(sorted_idx[0], int) else sorted_idx[0]]
        top_labels.append(label)
        if top_k > 1:
            top_list = []
            for idx in sorted_idx[:top_k]:
                lbl = id2label[str(idx) if isinstance(idx, int) else idx]
                top_list.append((lbl, float(prob_i[idx])))
            top_k_probs.append(top_list)
        else:
            top_k_probs.append([(label, float(prob_i[sorted_idx[0]]))])
    return top_labels, top_k_probs


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax for 2D array."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def predict_impact(
    model,
    model_type: str,
    feature_columns: List[str],
    event_labels: List[str],
) -> List[float]:
    """Predict impact given a list of event labels.

    Creates a one‑hot encoded feature matrix and feeds it through the
    impact model to obtain expected returns.
    """
    # Create one‑hot encoded DataFrame
    X = pd.get_dummies(event_labels, prefix="event")
    # Ensure all expected columns are present
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]
    if model_type == "lightgbm":
        preds = model.predict(X)
    else:
        preds = model.predict(X)
    return preds.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict event type and return impact for news headlines")
    parser.add_argument("--news_file", type=str, required=True, help="CSV file containing timestamp, ticker and headline columns")
    parser.add_argument("--event_model", type=str, required=True, help="Directory of the fine‑tuned event classifier")
    parser.add_argument("--impact_model", type=str, required=True, help="Path to the trained impact model (pkl or txt)")
    parser.add_argument("--top_k", type=int, default=1, help="Return top_k event types with probabilities for each headline")
    args = parser.parse_args()

    news_df = pd.read_csv(args.news_file)
    if not {"timestamp", "ticker", "headline"}.issubset(news_df.columns):
        raise ValueError("News file must contain 'timestamp', 'ticker' and 'headline' columns")
    # Normalize headlines
    headlines = normalize_batch(news_df["headline"].astype(str))

    # Load models
    tokenizer, event_model, id2label = load_event_classifier(args.event_model)
    impact_model, feature_columns, model_type, event_effects = load_impact_model(args.impact_model)

    # Predict event types
    labels, label_probs = predict_events(tokenizer, event_model, headlines, top_k=args.top_k)
    # Predict impact
    impacts = predict_impact(impact_model, model_type, feature_columns, labels)

    # Construct reasons
    reasons = []
    for lbl in labels:
        avg_effect = event_effects.get(lbl)
        if avg_effect is not None:
            direction = "positive" if avg_effect >= 0 else "negative"
            reasons.append(
                f"Historically, {lbl.replace('_', ' ')} events have a {direction} average return of {avg_effect:.4f}"
            )
        else:
            reasons.append(f"No historical data for {lbl} events; prediction based on model only")

    result_df = news_df[["timestamp", "ticker", "headline"]].copy()
    result_df["predicted_event"] = labels
    result_df["predicted_return"] = impacts
    result_df["reason"] = reasons

    # Display top_k probabilities if requested
    if args.top_k > 1:
        # Create a column with stringified top_k probabilities
        prob_strings = []
        for probs_list in label_probs:
            prob_strings.append(
                ", ".join([f"{lbl}:{prob:.2f}" for lbl, prob in probs_list])
            )
        result_df["top_k"] = prob_strings

    # Print results
    pd.set_option("display.max_colwidth", None)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()