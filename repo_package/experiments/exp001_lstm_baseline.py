"""
experiments/exp001_lstm_baseline.py

LSTM Experiment 1: baseline architecture for square 5161.

Deliberately small and short-sequence to start (staged tuning, not a final
config): single LSTM layer, hidden size 32, sequence length 24 (4 hours) -
comfortably past the PACF cutoff around lag 11, but well short of the
confirmed 144-step (24h) periodicity, so a follow-up experiment testing
L=144 is a real, evidence-motivated next step rather than an assumption.

Trains on 2013-11-01 to 2013-12-08, validates on 2013-12-09 to 2013-12-15.
December 16-22 (test) is not touched.
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

SEED = 42
SQUARE_ID = 5161
SEQ_LEN = 24
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.0
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 5


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(project_root: Path) -> dict:
    set_seed(SEED)

    interim_dir = project_root / "data" / "interim"
    series = load_square_series(SQUARE_ID, interim_dir)
    train, val, _test = chronological_split(series)

    scaler = fit_standard_scaler(train.values)
    train_scaled = apply_scaler(train.values, scaler)
    val_scaled = apply_scaler(val.values, scaler)

    X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
    X_val, y_val = create_sequences(val_scaled, SEQ_LEN)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    model = LSTMForecaster(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    n_train = len(X_train_t)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    final_train_loss = None
    epochs_run = 0

    start = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
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
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        print(f"Epoch {epoch:3d}  train_loss={final_train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    training_time = time.perf_counter() - start
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_pred_scaled = model(X_val_t).numpy()
    val_pred_actual = invert_scaler(val_pred_scaled, scaler)
    val_true_actual = invert_scaler(y_val, scaler)

    metrics = evaluate_metrics(val_true_actual, val_pred_actual)

    result = {
        "experiment_id": "EXP-LSTM-001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "LSTM",
        "square_id": SQUARE_ID,
        "sequence_length": SEQ_LEN,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "epochs_run": epochs_run,
        "train_loss_final": round(final_train_loss, 5),
        "val_MAE": round(metrics["MAE"], 3),
        "val_RMSE": round(metrics["RMSE"], 3),
        "val_MAPE": round(metrics["MAPE"], 3),
        "training_time_seconds": round(training_time, 2),
    }

    log_path = project_root / "experiments" / "experiment_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    print("\n" + "=" * 60)
    for k, v in result.items():
        print(f"{k}: {v}")
    print(f"\nAppended to: {log_path}")

    return result
