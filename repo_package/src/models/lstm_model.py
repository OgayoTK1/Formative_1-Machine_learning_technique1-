"""
src/models/lstm_model.py

Single-layer LSTM forecaster. Final architecture selected via staged
tuning (experiments EXP-LSTM-001 through 007, see experiments/experiment_log.csv):
hidden_size=64, num_layers=1, dropout=0.0 outperformed both wider (128) and
deeper (2-layer) alternatives, and a sequence length of 24 (4 hours)
outperformed 144 (24 hours) - the recurrent gating mechanism did not
benefit from the longer window, unlike the TCN and Transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)
