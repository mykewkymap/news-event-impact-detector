"""
Streamlit application for the News Event Impact Detector.

This app provides a simple graphical interface to classify financial
news headlines into event categories and estimate the expected
return impact for each headline.  Users can upload a CSV file
containing news events, choose the trained models to use, and
explore the results interactively.

The app reuses the core functions from ``predict.py`` to load the
fine‑tuned event classifier and the impact regression model.  It then
normalizes the headlines, predicts the event type for each headline,
estimates the forward return, and constructs a qualitative reason
based on historical averages saved during training.

To run the app, make sure you have installed the dependencies in
``requirements.txt`` (including Streamlit), and execute:

```
streamlit run app.py
```

At a minimum you will need a directory with a trained event
classifier (e.g. ``models/event_classifier``) and a saved impact
model (e.g. ``models/impact_model.pkl``).  You can use the example
training scripts in this repository to create these artifacts.

"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.text_utils import normalize_batch
from predict import (
    load_event_classifier,
    load_impact_model,
    predict_events,
    predict_impact,
    softmax,
)


def main() -> None:
    st.set_page_config(page_title="News Event Impact Detector", layout="wide")
    st.title("📰 News Event Impact Detector")
    st.markdown(
        """
        Upload a CSV file of news events and classify each headline into a financial
        event type (e.g. earnings beat, guidance cut, litigation, FDA approval).
        The app then estimates the expected forward return based on the event
        type and displays a qualitative reason for each prediction.
        """
    )

    # Sidebar configuration
    st.sidebar.header("Settings")
    event_model_dir = st.sidebar.text_input(
        "Event classifier directory", value="models/event_classifier"
    )
    impact_model_path = st.sidebar.text_input(
        "Impact model path", value="models/impact_model.pkl"
    )
    top_k = st.sidebar.slider(
        "Number of event candidates to display (top_k)", min_value=1, max_value=3, value=1
    )

    uploaded_file = st.file_uploader(
        "Upload CSV with columns timestamp,ticker,headline", type=["csv"]
    )
    if uploaded_file is not None:
        # Read CSV
        try:
            news_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return

        required_cols = {"timestamp", "ticker", "headline"}
        if not required_cols.issubset(news_df.columns):
            st.error(
                f"CSV file must contain columns: {', '.join(required_cols)}"
            )
            return

        # Normalize headlines
        headlines = normalize_batch(news_df["headline"].astype(str))

        # Load models once per session
        @st.cache_data(show_spinner=False)
        def load_models(event_model_dir: str, impact_model_path: str):
            try:
                event_tokenizer, event_model, id2label = load_event_classifier(
                    event_model_dir
                )
            except Exception as e:
                st.error(f"Failed to load event classifier: {e}")
                return None, None, None, None, None, None, None
            try:
                impact_model, feature_columns, model_type, event_effects = load_impact_model(
                    impact_model_path
                )
            except Exception as e:
                st.error(f"Failed to load impact model: {e}")
                return None, None, None, None, None, None, None
            return (
                event_tokenizer,
                event_model,
                id2label,
                impact_model,
                feature_columns,
                model_type,
                event_effects,
            )

        models = load_models(event_model_dir, impact_model_path)
        if any(m is None for m in models):
            return
        (
            event_tokenizer,
            event_model,
            id2label,
            impact_model,
            feature_columns,
            model_type,
            event_effects,
        ) = models

        # Prediction
        with st.spinner("Predicting events and returns..."):
            labels, label_probs = predict_events(
                event_tokenizer, event_model, headlines, top_k=top_k
            )
            impacts = predict_impact(
                impact_model, model_type, feature_columns, labels
            )

        # Construct reasons
        reasons: List[str] = []
        for lbl in labels:
            avg_effect = event_effects.get(lbl)
            if avg_effect is not None:
                direction = "positive" if avg_effect >= 0 else "negative"
                reasons.append(
                    f"Historically, {lbl.replace('_', ' ')} events have a {direction} average return of {avg_effect:.4f}"
                )
            else:
                reasons.append(
                    f"No historical data for {lbl} events; prediction based on model only"
                )

        # Assemble results DataFrame
        result_df = news_df[["timestamp", "ticker", "headline"]].copy()
        result_df["predicted_event"] = labels
        result_df["predicted_return"] = impacts
        result_df["reason"] = reasons
        if top_k > 1:
            prob_strings = []
            for probs_list in label_probs:
                prob_strings.append(
                    ", ".join([f"{lbl}:{prob:.2f}" for lbl, prob in probs_list])
                )
            result_df["top_k_probs"] = prob_strings

        # Display results
        st.subheader("Predictions")
        st.dataframe(result_df, use_container_width=True)

        # Allow downloading results
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="event_impact_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("Upload a CSV file to start")


if __name__ == "__main__":
    main()