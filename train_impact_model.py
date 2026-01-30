#!/usr/bin/env python3
"""
Train a model to predict the return impact of classified news events.

This script expects a CSV dataset produced by
``data/prepare_impact_dataset.py`` containing at least the columns
``event_type`` and ``return``.  It encodes the event type into
numerical features (one-hot) and trains a regression model (default
RandomForestRegressor) to map events to forward returns.  The trained
model and metadata are saved to disk for later inference.

Example usage:

```
python train_impact_model.py \
  --input_file data/impact_dataset.csv \
  --output_model models/impact_model.pkl
```

You can customise the algorithm via the ``--model`` argument.  Options
include ``random_forest``, ``linear_regression``, and ``lightgbm`` if
LightGBM is installed.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def train_random_forest(X_train, y_train) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_linear_regression(X_train, y_train) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train) -> lgb.Booster:
    train_ds = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42,
    }
    model = lgb.train(params, train_ds, num_boost_round=200)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a return impact model using event type features.")
    parser.add_argument("--input_file", type=str, required=True, help="CSV file with event_type and return columns")
    parser.add_argument("--output_model", type=str, required=True, help="Path to save the trained model")
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "linear_regression", "lightgbm"],
        help="Model type to train",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    if "event_type" not in df.columns or "return" not in df.columns:
        raise ValueError("Input file must contain 'event_type' and 'return' columns")
    # Drop rows with missing values
    df = df.dropna(subset=["event_type", "return"])

    # Encode event types using one-hot encoding
    X = pd.get_dummies(df["event_type"], prefix="event")
    y = df["return"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if args.model == "random_forest":
        model = train_random_forest(X_train, y_train)
    elif args.model == "linear_regression":
        model = train_linear_regression(X_train, y_train)
    elif args.model == "lightgbm":
        if not HAS_LIGHTGBM:
            raise RuntimeError("LightGBM is not installed. Please install lightgbm or choose another model.")
        model = train_lightgbm(X_train, y_train)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Evaluate model on test set
    if args.model == "lightgbm":
        preds = model.predict(X_test)
    else:
        preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Test RMSE: {rmse:.6f}")

    # Save model and metadata (event columns) using joblib or LightGBM's save method
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    if args.model == "lightgbm":
        model.save_model(args.output_model)
        # Save feature names separately
        metadata = {"feature_columns": X.columns.tolist(), "model_type": args.model}
        with open(args.output_model + ".meta.json", "w") as f:
            json.dump(metadata, f, indent=2)
    else:
        joblib.dump({"model": model, "feature_columns": X.columns.tolist(), "model_type": args.model}, args.output_model)
    print(f"Saved impact model to {args.output_model}")

    # Compute and save average impact per event type to aid explanation
    event_effects = df.groupby("event_type")["return"].mean().to_dict()
    effects_path = args.output_model + ".effects.json"
    with open(effects_path, "w") as f:
        json.dump(event_effects, f, indent=2)
    print(f"Saved average event effects to {effects_path}")


if __name__ == "__main__":
    main()