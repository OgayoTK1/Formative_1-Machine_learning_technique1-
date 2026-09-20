"""
src/models/transformer_model.py

Compact self-attention encoder for one-step-ahead forecasting. Genuinely
different mechanism from both the LSTM (recurrence) and TCN (dilated
convolution): attention can reference any position in the input window
directly, including lag 144, without needing recurrent gating or a deep
dilation stack to physically reach it. This is the specific architectural
property this model was selected to test.

Deliberately small (d_model=32, 2 heads, 2 layers) given the CPU-only,
2-vCPU runtime documented at the start of this project - flagged from the
outset as the highest computational risk of the three models.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerForecaster(nn.Module):
    def __init__(self, d_model: int = 32, nhead: int = 2, num_layers: int = 2,
                 dim_feedforward: int = 64, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1). No causal mask needed: the input window is
        # purely historical (t-L+1..t), the target x(t+1) is never inside it,
        # so full (non-causal) self-attention over the window has no leakage.
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)
