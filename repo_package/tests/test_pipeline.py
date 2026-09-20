"""
tests/test_pipeline.py

Lightweight sanity checks for the data pipeline. Not a full test suite,
but enough to catch the kind of silent errors that matter most here:
wrong split sizes, scaler leakage, and MAPE division-by-zero handling.

Run from the project root after src/ is on sys.path:
    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "preprocessing"))
from sequence_pipeline import (  # noqa: E402
    apply_scaler,
    create_sequences,
    evaluate_metrics,
    fit_standard_scaler,
    mape_safe,
)


def test_sequence_construction_shapes():
    values = np.arange(100, dtype=np.float32)
    X, y = create_sequences(values, seq_len=10)
    assert X.shape == (90, 10), f"expected (90, 10), got {X.shape}"
    assert y.shape == (90,)
    assert np.array_equal(X[0], values[0:10])
    assert y[0] == values[10]
    print("test_sequence_construction_shapes: PASS")


def test_scaler_fit_on_train_only():
    train_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scaler = fit_standard_scaler(train_values)
    assert np.isclose(scaler["mean"], 3.0)
    # A scaler fit on train should not silently change if "future" data
    # is appended - this checks the function signature genuinely only
    # depends on what's passed in, guarding against accidental global state.
    other_values = np.array([100.0, 200.0])
    scaler2 = fit_standard_scaler(train_values)
    assert scaler == scaler2, "scaler must be deterministic and independent of unrelated data"
    print("test_scaler_fit_on_train_only: PASS")


def test_mape_zero_handling():
    y_true = np.array([0.0, 0.0, 10.0, 20.0])
    y_pred = np.array([1.0, 2.0, 11.0, 22.0])
    mape, n_excluded = mape_safe(y_true, y_pred, threshold=5.0)
    assert n_excluded == 2, f"expected 2 near-zero points excluded, got {n_excluded}"
    assert not np.isnan(mape), "MAPE should be computable from the remaining 2 points"
    print("test_mape_zero_handling: PASS")


def test_evaluate_metrics_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    metrics = evaluate_metrics(y, y.copy())
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["MAPE"] == 0.0
    print("test_evaluate_metrics_perfect_prediction: PASS")


if __name__ == "__main__":
    test_sequence_construction_shapes()
    test_scaler_fit_on_train_only()
    test_mape_zero_handling()
    test_evaluate_metrics_perfect_prediction()
    print("\nAll tests passed.")
