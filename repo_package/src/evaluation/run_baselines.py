"""
src/evaluation/run_baselines.py

Evaluate persistence and seasonal-naive (144-lag) baselines on the
VALIDATION period (Dec 9-15, 2013) for the three highest-traffic squares
(5161, 5059, 5259). December 16-22 (test) is deliberately not touched here.

These baselines answer: does a complex model actually learn useful
structure beyond simple temporal repetition?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sequence_pipeline import (
    SEASONAL_LAG,
    chronological_split,
    evaluate_metrics,
    load_square_series,
)

SQUARE_IDS = [5161, 5059, 5259]


def run_baselines(interim_dir: Path, results_dir: Path) -> pd.DataFrame:
    records = []

    for sq in SQUARE_IDS:
        series = load_square_series(sq, interim_dir)
        train, val, _test = chronological_split(series)

        combined = pd.concat([train, val])
        values = combined.values
        val_start_idx = len(train)

        # Persistence: pred(t+1) = actual(t)
        y_true_persist = values[val_start_idx:]
        y_pred_persist = values[val_start_idx - 1: -1]
        metrics_persist = evaluate_metrics(y_true_persist, y_pred_persist)
        records.append({"square_id": sq, "model": "persistence", **metrics_persist})

        # Seasonal naive: pred(t+1) = actual(t+1-144), using train history for early val points
        y_true_seasonal = values[val_start_idx:]
        y_pred_seasonal = values[val_start_idx - SEASONAL_LAG: len(values) - SEASONAL_LAG]
        metrics_seasonal = evaluate_metrics(y_true_seasonal, y_pred_seasonal)
        records.append({"square_id": sq, "model": "seasonal_naive_144", **metrics_seasonal})

    results_df = pd.DataFrame(records)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "baseline_validation_metrics.csv"
    results_df.to_csv(out_path, index=False)

    print(results_df.to_string(index=False))
    print(f"\nSaved to: {out_path}")

    return results_df
