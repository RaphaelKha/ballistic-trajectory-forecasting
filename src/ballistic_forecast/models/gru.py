"""Seq2seq GRU forecaster: the original model of this project, cleaned up.

Encoder GRU reads the observed track; an autoregressive decoder emits one value
per future step. With ``output="accel"`` (default) it is an acceleration that is
integrated twice from the measured velocity; with ``output="delta"`` (the
original design, kept for the ablation) it is a free position displacement.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ACC_SCALE, OUTPUT_MODES, initial_velocity, kinematic_features


class GRUForecaster(nn.Module):
    def __init__(self, hidden_dim: int = 256, pred_len: int = 30, output: str = "accel"):
        super().__init__()
        assert output in OUTPUT_MODES, output
        self.pred_len = pred_len
        self.output = output
        self.input_proj = nn.Linear(4, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, T, 2) relative coords -> (B, H, 2) relative coords."""
        feats = kinematic_features(obs)
        _, hidden = self.gru(F.gelu(self.input_proj(feats)))

        pos = obs[:, -1:, :]
        if self.output == "accel":
            vel = initial_velocity(obs)[:, None]
            step = torch.cat([pos, vel * 10.0], dim=-1)
        else:
            step = feats[:, -1:, :]
        preds = []
        for _ in range(self.pred_len):
            out, hidden = self.gru(F.gelu(self.input_proj(step)), hidden)
            if self.output == "accel":
                vel = vel + self.output_proj(out) * ACC_SCALE
            else:
                vel = self.output_proj(out)
            pos = pos + vel
            preds.append(pos)
            step = torch.cat([pos, vel * 10.0], dim=-1)
        return torch.cat(preds, dim=1)
