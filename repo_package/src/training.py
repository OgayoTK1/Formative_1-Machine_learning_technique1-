"""
src/training.py

Model-agnostic training and evaluation harness. Any nn.Module that accepts
input shaped (batch, seq_len, 1) and returns (batch,) can be trained with
this function, so the same code serves LSTM, TCN, and Transformer without
duplication.

Logs every run to experiments/experiment_log.csv with a fixed superset of
columns; architecture-specific hyperparameters not applicable to a given
model are left blank rather than causing a schema mismatch.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sequence_pipeline import (
    apply_scaler,
    chronological_split,
    create_sequences,
    evaluate_metrics,
    fit_standard_scaler,
    invert_scaler,
    load_square_series,
)

LOG_COLUMNS = [
    "experiment_id", "timestamp", "model", "square_id", "sequence_length",
    "hidden_size", "num_layers", "channels", "kernel_size", "dilations",
    "dropout", "learning_rate", "batch_size", "optimizer", "epochs_run",
    "train_loss_final", "val_MAE", "val_RMSE", "val_MAPE",
    "training_time_seconds", "notes",
]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_and_evaluate(
    model: nn.Module,
    square_id: int,
    seq_len: int,
    project_root: Path,
    experiment_id: str,
    model_name: str,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 50,
    patience: int = 5,
    seed: int = 42,
    notes: str = "",
    architecture_fields: dict | None = None,
) -> dict:
    set_seed(seed)
    architecture_fields = architecture_fields or {}

    interim_dir = project_root / "data" / "interim"
    series = load_square_series(square_id, interim_dir)
    train, val, _test = chronological_split(series)

    scaler = fit_standard_scaler(train.values)
    train_scaled = apply_scaler(train.values, scaler)
    val_scaled = apply_scaler(val.values, scaler)

    X_train, y_train = create_sequences(train_scaled, seq_len)
    X_val, y_val = create_sequences(val_scaled, seq_len)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    n_train = len(X_train_t)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    final_train_loss = None
    epochs_run = 0
    best_state = None

    start = time.perf_counter()
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
        final_train_loss = epoch_loss / n_train
        epochs_run = epoch

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        print(f"Epoch {epoch:3d}  train_loss={final_train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    training_time = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_pred_scaled = model(X_val_t).numpy()
    val_pred_actual = invert_scaler(val_pred_scaled, scaler)
    val_true_actual = invert_scaler(y_val, scaler)

    metrics = evaluate_metrics(val_true_actual, val_pred_actual)

    result = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "square_id": square_id,
        "sequence_length": seq_len,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "epochs_run": epochs_run,
        "train_loss_final": round(final_train_loss, 5),
        "val_MAE": round(metrics["MAE"], 3),
        "val_RMSE": round(metrics["RMSE"], 3),
        "val_MAPE": round(metrics["MAPE"], 3),
        "training_time_seconds": round(training_time, 2),
        "notes": notes,
    }
    result.update(architecture_fields)

    log_path = project_root / "experiments" / "experiment_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore", restval="")
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    print("\n" + "=" * 60)
    for k, v in result.items():
        print(f"{k}: {v}")
    print(f"\nAppended to: {log_path}")

    return result
