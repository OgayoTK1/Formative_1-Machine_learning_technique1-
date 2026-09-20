"""
src/models/tcn_model.py

Dilated causal Temporal Convolutional Network (Bai et al., 2018 style),
adapted for scalar one-step-ahead forecasting.

Genuinely different mechanism from the LSTM: parallel dilated convolutions
instead of sequential recurrence. Dilation doubles per layer (1, 2, 4, 8, ...),
so a 4-layer stack with kernel_size=3 has a receptive field of
1 + 2*(kernel_size-1)*(1+2+4+8) = 61 steps, comfortably covering the
seq_len=24 window used as the LSTM's final configuration for direct
comparison, while remaining extensible toward the 144-step daily cycle
by adding layers if a later experiment calls for it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Removes the extra right-side padding used to keep convolutions causal."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(nn.Module):
    def __init__(self, num_channels: list[int], kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        for i, ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = 1 if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_ch, ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x: (batch, seq_len, 1) -> (batch, 1, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        out = self.network(x)
        last = out[:, :, -1]
        return self.fc(last).squeeze(-1)
