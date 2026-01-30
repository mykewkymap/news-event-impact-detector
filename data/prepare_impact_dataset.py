#!/usr/bin/env python3
"""
Prepare the dataset for training the return impact model.

Given a CSV file containing news events with timestamps, tickers and
headlines, this script computes forward returns over a specified
horizon for each event using historical price data from Yahoo!
Finance.  The resulting dataset includes the original headline,
normalized event type (if present), and the realised return over the
forward horizon.  This dataset can then be used to train the impact
model (regressor or classifier) that maps events to price impact.

The input CSV must contain at least the following columns:

```
timestamp,ticker,headline
```

It may also include a manually annotated ``event_type`` column for
training the event classifier.  The script will preserve this column
if present.  The ``timestamp`` column should be in ISO 8601 format
(YYYY-MM-DD HH:MM:SS) and should correspond to the date/time the
headline was published.  For daily granularity, the time component
can be omitted.

Example usage:

```
python data/prepare_impact_dataset.py \
  --events_file data/events.csv \
  --days 5 \
  --output data/impact_dataset.csv
```
"""

from __future__ import annotations

import argparse
import datetime
from typing import List

import pandas as pd
import yfinance as yf

from utils.text_utils import normalize_batch


def compute_forward_returns(
    events_df: pd.DataFrame, days: int, price_data: pd.DataFrame
) -> pd.Series:
    """Compute the forward return for each event.

    Parameters
    ----------
    events_df : pandas.DataFrame
        DataFrame containing columns ``date`` and ``ticker``.
    days : int
        Number of trading days to look ahead when computing the return.
    price_data : pandas.DataFrame
        Multi‑indexed DataFrame of OHLCV data from yfinance with tickers as
        the top level and fields as the second level.

    Returns
    -------
    pandas.Series
        Forward return for each row in ``events_df``.  Rows for which
        price data is missing will contain NaN.
    """
    returns = []
    for idx, row in events_df.iterrows():
        ticker = row["ticker"]
        event_date = row["date"]
        try:
            # Extract closing price series for this ticker
            closes = price_data[ticker]["Close"].dropna()
        except KeyError:
            returns.append(float('nan'))
            continue
        # Find price on event date and price days ahead
        if event_date not in closes.index:
            # If event occurs on non‑trading day, align to next trading day
            future_dates = closes.index[closes.index >= event_date]
            if future_dates.empty:
                returns.append(float('nan'))
                continue
            current_date = future_dates[0]
        else:
            current_date = event_date
        future_index = closes.index.get_loc(current_date)
        target_index = future_index + days
        if target_index >= len(closes):
            returns.append(float('nan'))
            continue
        current_price = closes.iloc[future_index]
        future_price = closes.iloc[target_index]
        if current_price == 0:
            returns.append(float('nan'))
        else:
            returns.append(future_price / current_price - 1.0)
    return pd.Series(returns, index=events_df.index, name=f"return_{days}d")


def download_price_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download historical price data for the specified tickers.

    Parameters
    ----------
    tickers : list of str
        List of symbols to download.
    start : str
        Start date (inclusive) in YYYY-MM-DD format.
    end : str
        End date (exclusive) in YYYY-MM-DD format.

    Returns
    -------
    pandas.DataFrame
        Multi‑indexed DataFrame with tickers as top level and OHLCV
        fields as second level.
    """
    price_data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=True,
    )
    return price_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare impact dataset by computing forward returns.")
    parser.add_argument("--events_file", type=str, required=True, help="CSV file with timestamp, ticker and headline columns")
    parser.add_argument("--days", type=int, default=5, help="Number of trading days ahead to compute return")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file to save the dataset")
    args = parser.parse_args()

    # Load events
    events = pd.read_csv(args.events_file)
    if "timestamp" not in events.columns or "ticker" not in events.columns or "headline" not in events.columns:
        raise ValueError("Input file must contain 'timestamp', 'ticker' and 'headline' columns")

    # Convert timestamp to datetime and extract date (without time)
    events["date"] = pd.to_datetime(events["timestamp"]).dt.date
    events["ticker"] = events["ticker"].str.upper().str.strip()
    # Normalize headlines
    events["headline_normalized"] = normalize_batch(events["headline"].astype(str))

    # Determine the required date range for price data
    min_date = events["date"].min()
    max_date = events["date"].max()
    # We need prices at least ``days`` after the last event date
    max_future_date = max_date + datetime.timedelta(days=args.days * 2)
    # Convert to strings for yfinance
    start_str = (min_date - datetime.timedelta(days=5)).strftime("%Y-%m-%d")
    end_str = max_future_date.strftime("%Y-%m-%d")
    tickers = sorted(events["ticker"].unique().tolist())
    print(f"Downloading price data for {len(tickers)} tickers from {start_str} to {end_str}…")
    price_data = download_price_data(tickers, start_str, end_str)

    # Compute forward returns
    events["return"] = compute_forward_returns(events, args.days, price_data)
    # Drop rows where return is not available
    dataset = events.dropna(subset=["return"])

    # Save the dataset
    dataset.to_csv(args.output, index=False)
    print(f"Saved impact dataset with {len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()