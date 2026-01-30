# News Event Impact Detector

This repository implements a **news‑driven event detection and return impact system** for equities.  It combines
state‑of‑the‑art natural language processing with classic regression to answer a simple question:

> *Given a new headline about a company, what kind of event does it describe and is the news likely to move the stock price up or down over the next few days?*

Unlike traditional stock price forecasting, this project focuses on **identifying actionable events** and estimating their directional impact.  It classifies headlines into a handful of canonical event types (earnings beats, guidance cuts, litigation, FDA approvals, etc.), then maps these events to expected returns using a separate model.  The underlying technique is inspired by recent research showing that **soft information extracted from corporate press releases is just as informative as the numerical earnings surprise, with FinBERT yielding the highest predictive power for textual features**【2315255087894†L48-L55】.  Event detection and classification are critical tasks for market analysis【638170061999060†L61-L88】 because positive or negative events can trigger substantial buying or selling, and NLP techniques enable investors to process large volumes of news more efficiently【638170061999060†L122-L131】.

## Features

- **Event classification**: A transformer model fine‑tuned on financial news identifies the type of event described by each headline.  We use FinBERT as the base model because it is pre‑trained on financial text and captures domain‑specific nuances【248260085656625†L68-L76】.
- **Impact modelling**: A separate regression model learns the typical forward return associated with each event type.  It can be a random forest, linear regression or LightGBM.
- **Data preparation**: Scripts to compute forward returns from raw event logs and historical price data.
- **Explainability**: The system outputs both the predicted event type and a concise reason based on historical average effects (e.g., “Historically, FDA approval events have a positive average return of 3%”).

## Repository structure

```
news-event-impact-detector/
├── data/
│   └── prepare_impact_dataset.py    # Compute forward returns for events
├── utils/
│   └── text_utils.py               # Helpers for text normalization
├── train_event_classifier.py       # Fine‑tune FinBERT for event classification
├── train_impact_model.py           # Train regression model to predict returns from event types
├── predict.py                      # End‑to‑end inference: classify events and estimate impact
├── models/                         # Saved models and metadata (created after training)
├── requirements.txt                # Python dependencies
└── README.md                       # This documentation
```

## Getting started

### 1. Install dependencies

Use Python 3.8 or later and install the required packages:

```bash
pip install -r requirements.txt
```

Note: Fine‑tuning FinBERT requires PyTorch and the Hugging Face `transformers` library.  The repository does not include model weights; the scripts will download them automatically from the Hugging Face hub when run.

### 2. Collect and label event data

To train the event classifier you need a CSV file (`event_classification_train.csv`) with at least two columns:

- `headline`: the news headline text.
- `event_type`: one of the event categories you wish to classify (e.g., `earnings_beat`, `guidance_cut`, `litigation`, `fda_approval`).

You can create this dataset manually or by combining news feeds with known corporate events.  Research emphasises that the ability to quickly classify financial events can be the difference between success and failure in trading【638170061999060†L61-L88】.

#### Train the event classifier

```bash
python train_event_classifier.py \
  --input_file event_classification_train.csv \
  --model_name ProsusAI/finbert \
  --output_dir models/event_classifier \
  --epochs 3
```

This script fine‑tunes FinBERT on your labelled headlines and saves the model and label mapping to `models/event_classifier`.

### 3. Prepare the impact dataset

Next, compute forward returns for a set of events.  Suppose you have a CSV (`events.csv`) with columns `timestamp`, `ticker` and `headline` (and optionally `event_type` if manually annotated).  Use `prepare_impact_dataset.py` to merge this with historical price data:

```bash
python data/prepare_impact_dataset.py \
  --events_file events.csv \
  --days 5 \
  --output data/impact_dataset.csv
```

This script downloads daily price data from Yahoo! Finance and computes the percentage return after a specified number of trading days.  The resulting file includes a `return` column for each event.  You can adjust the `--days` argument (e.g. 5 or 10) to match your horizon.

### 4. Train the impact model

Once you have returns and event types, train a regression model that maps event categories to future returns:

```bash
python train_impact_model.py \
  --input_file data/impact_dataset.csv \
  --output_model models/impact_model.pkl \
  --model random_forest
```

This script encodes event types as one‑hot vectors, trains the chosen regression model, and saves it along with the list of feature columns and the average historical return per event type.  You can choose between `random_forest`, `linear_regression` and `lightgbm` (requires LightGBM installed).

### 5. Make predictions

To classify new headlines and estimate their impact, prepare a CSV (`new_events.csv`) with columns `timestamp`, `ticker` and `headline`, then run:

```bash
python predict.py \
  --news_file new_events.csv \
  --event_model models/event_classifier \
  --impact_model models/impact_model.pkl \
  --top_k 1
```

The script outputs a table of news events with the predicted event type, expected return and a qualitative reason based on historical averages.  If you specify `--top_k > 1`, it will also show the top event candidates with their probabilities.

## 6. Explore results with the Streamlit app

For a graphical interface, you can run the included **Streamlit** app.  It allows you to upload a CSV of news events, choose your trained models, and view predictions in an interactive table.

```bash
streamlit run app.py
```

The app expects the same `timestamp`, `ticker` and `headline` columns as `predict.py`.  It loads the event classifier and impact model from the paths specified in the sidebar (defaulting to `models/event_classifier` and `models/impact_model.pkl`).  After processing your file it displays the predicted event, expected return and qualitative reason for each headline.  You can also download the results as a CSV.

## How it works

1. **Event classification** – A transformer model (FinBERT by default) encodes each headline and predicts the event type.  FinBERT is pre‑trained on a large corpus of financial communication【248260085656625†L68-L76】 and excels at capturing domain‑specific sentiment and tone.
2. **Return impact modelling** – A separate regression model learns the typical forward return for each event category.  Research shows that soft textual information from corporate announcements can be as informative as hard numerical surprises when explaining stock reactions【2315255087894†L48-L55】.
3. **Explainability** – During prediction we include a simple reason such as “Historically, earnings beat events have a positive average return of 2.5%”.  This helps users interpret the model output.

## Extending the project

- **More event types** – Add additional labels (e.g., mergers, product launches, share buybacks) to the training data.  The model will automatically expand to recognise them.
- **Better models** – Swap the regression model for something more sophisticated (e.g., LightGBM or neural networks) or augment the features with sentiment scores.  The classification article notes that DistilBERT and other models can outperform traditional classifiers on financial event classification tasks【638170061999060†L165-L195】.
- **Streaming inference** – Wrap `predict.py` into a web service or integrate with a streaming news API to generate alerts in real time.
- **Joint training** – For research purposes, you can jointly optimise the event classifier and impact model end‑to‑end using multi‑task learning.

## Limitations and disclaimer

This project is for educational purposes and **does not constitute financial advice**.  The impact model is based on historical averages and does not guarantee future performance.  Proper backtesting and risk management are essential before deploying any trading strategy.

## References

- FinBERT introduces a transformer pre‑trained on a large financial communication corpus【248260085656625†L68-L76】.
- Research on press releases shows that soft information can be as informative as hard numerical surprises, with FinBERT providing the strongest textual features【2315255087894†L48-L55】.
- Articles on financial event classification emphasise the importance of quickly identifying positive or negative events and outline key tasks such as event detection, classification and summarisation【638170061999060†L61-L88】【638170061999060†L122-L131】.