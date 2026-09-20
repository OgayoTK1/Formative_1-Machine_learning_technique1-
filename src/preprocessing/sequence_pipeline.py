"""
src/preprocessing/sequence_pipeline.py

Shared, leakage-safe data pipeline for one-step-ahead forecasting on a
single geographical square's Internet traffic series.

Split boundaries (chronological, no shuffling):
    train:      2013-11-01 00:00 - 2013-12-08 23:50  (38 days)
    validation: 2013-12-09 00:00 - 2013-12-15 23:50  (7 days)
    test:       2013-12-16 00:00 - 2013-12-22 23:50  (7 days, required period)

2013-12-23 onward is deliberately excluded from every split: it postdates
the required test period, and using it for training would break strict
chronological ordering even though it would not directly leak into the
Dec 16-22 predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_START = "2013-11-01"
TRAIN_END = "2013-12-08 23:50:00+01:00"
VAL_START = "2013-12-09 00:00:00+01:00"
VAL_END = "2013-12-15 23:50:00+01:00"
TEST_START = "2013-12-16 00:00:00+01:00"
TEST_END = "2013-12-22 23:50:00+01:00"

SEASONAL_LAG = 144  # 24 hours at 10-minute resolution, confirmed via ACF local-peak check


def load_square_series(square_id: int, interim_dir: Path) -> pd.Series:
    """Load one square's full observation-period series, indexed by local (Europe/Rome) timestamp."""
    files = sorted(interim_dir.glob("*.parquet"))
    frames = [pd.read_parquet(p, filters=[("square_id", "=", square_id)]) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.tz_convert("Europe/Rome")
    df = df.sort_values("timestamp").set_index("timestamp")
    series = df["internet"].astype("float32")

    expected = pd.date_range(series.index.min(), series.index.max(), freq="10min", tz="Europe/Rome")
    missing = expected.difference(series.index)
    if len(missing) > 0:
        raise ValueError(f"Square {square_id}: {len(missing)} missing intervals, e.g. {missing[:5].tolist()}")

    return series


def chronological_split(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    train = series[TRAIN_START:TRAIN_END]
    val = series[VAL_START:VAL_END]
    test = series[TEST_START:TEST_END]
    return train, val, test


def create_sequences(values: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """One-step-ahead sequences: X[i] = values[i:i+seq_len], y[i] = values[i+seq_len]."""
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X), np.array(y)


def fit_standard_scaler(train_values: np.ndarray) -> dict:
    """Fit a standardization scaler on TRAINING DATA ONLY. Never call this on val/test."""
    return {"mean": float(np.mean(train_values)), "std": float(np.std(train_values))}


def apply_scaler(values: np.ndarray, scaler: dict) -> np.ndarray:
    return (values - scaler["mean"]) / scaler["std"]


def invert_scaler(values: np.ndarray, scaler: dict) -> np.ndarray:
    return values * scaler["std"] + scaler["mean"]


def mape_safe(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> tuple[float, int]:
    """
    MAPE computed only over points where |y_true| >= threshold, to avoid
    division blowup near zero/near-zero actual traffic. Returns (mape, n_excluded)
    so the exclusion is visible rather than silent.
    """
    mask = np.abs(y_true) >= threshold
    n_excluded = int((~mask).sum())
    if mask.sum() == 0:
        return float("nan"), n_excluded
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return mape, n_excluded


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, mape_threshold: float = 10.0) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape, n_excluded = mape_safe(y_true, y_pred, mape_threshold)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "mape_excluded_count": n_excluded,
        "mape_threshold": mape_threshold,
    }
