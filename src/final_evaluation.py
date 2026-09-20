"""
src/final_evaluation.py

Final training and test-period evaluation for the required Dec 16-22
comparison across three geographical areas and three models.

CORRECTED APPROACH: trains on TRAIN only (Nov 1 - Dec 8) with early
stopping against VALIDATION (Dec 9 - Dec 15), restoring best-val-loss
weights - the exact same procedure used during hyperparameter tuning,
now re-run once per square with each model's final chosen architecture.

An earlier version of this function trained on train+validation combined
for a fixed epoch count (whatever early stopping selected during tuning).
That produced a badly mis-converged Transformer on 2 of 3 squares (test
MAPE 26.1% and 14.5% vs a tuning-phase validation MAPE of 7.81%), because
the epoch count selected against one dataset's loss landscape does not
reliably transfer to a differently-composed one. Reverting to genuine
early stopping removes that fragility; the cost is not using the extra
7 days of validation-period data for final training, which is an
explicit, stated trade-off favoring a validated stopping point over
marginally more training data.

Test-period evaluation still uses true historical values as input at
every step (including earlier test-period actuals once the prediction
point has moved past them) - standard one-step-ahead protocol, not
leakage.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from sequence_pipeline import (
    apply_scaler,
    chronological_split,
    create_sequences,
    evaluate_metrics,
    fit_standard_scaler,
    load_square_series,
)
from training import set_seed


def train_final_and_test(
    model: nn.Module,
    square_id: int,
    seq_len: int,
    project_root: Path,
    model_name: str,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 50,
    patience: int = 5,
    seed: int = 42,
) -> dict:
    set_seed(seed)

    interim_dir = project_root / "data" / "interim"
    series = load_square_series(square_id, interim_dir)
    train, val, test = chronological_split(series)
    all_data = pd.concat([train, val, test])

    scaler = fit_standard_scaler(train.values)
    train_scaled = apply_scaler(train.values, scaler)
    val_scaled = apply_scaler(val.values, scaler)
    all_scaled = apply_scaler(all_data.values, scaler)

    X_train, y_train = create_sequences(train_scaled, seq_len)
    X_val, y_val = create_sequences(val_scaled, seq_len)
    X_all, y_all = create_sequences(all_scaled, seq_len)

    n_test = len(test)
    X_test = X_all[-n_test:]
    y_test_scaled = y_all[-n_test:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    n_train = len(X_train_t)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None
    epochs_run = 0

    train_start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epochs_run = epoch

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        print(f"Epoch {epoch:3d}  train_loss={epoch_loss / n_train:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    training_time = time.perf_counter() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    inference_start = time.perf_counter()
    with torch.no_grad():
        pred_scaled = model(X_test_t).numpy()
    inference_time = time.perf_counter() - inference_start

    pred_actual = pred_scaled * scaler["std"] + scaler["mean"]
    true_actual = y_test_scaled * scaler["std"] + scaler["mean"]

    metrics = evaluate_metrics(true_actual, pred_actual)

    pred_dir = project_root / "results" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({
        "timestamp": test.index,
        "actual": true_actual,
        "predicted": pred_actual,
    })
    pred_path = pred_dir / f"{model_name}_{square_id}.csv"
    pred_df.to_csv(pred_path, index=False)

    result = {
        "model": model_name,
        "square_id": square_id,
        "sequence_length": seq_len,
        "epochs_run": epochs_run,
        "MAE": round(metrics["MAE"], 3),
        "RMSE": round(metrics["RMSE"], 3),
        "MAPE": round(metrics["MAPE"], 3),
        "mape_excluded_count": metrics["mape_excluded_count"],
        "training_time_seconds": round(training_time, 2),
        "inference_time_seconds": round(inference_time, 4),
        "predictions_path": str(pred_path),
    }

    print("\n" + "=" * 60)
    for k, v in result.items():
        print(f"{k}: {v}")

    return result
