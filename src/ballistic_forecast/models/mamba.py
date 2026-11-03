"""Mamba (selective state-space model) forecaster in pure PyTorch.

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
State Spaces" (2023). The official ``mamba_ssm`` package needs CUDA kernels;
this implementation runs the same S6 recurrence with a plain loop over time so
it works on CPU / Apple Silicon. Our sequences are 60 tokens long, so the loop
is cheap.

Forecasting scheme (non-autoregressive):

    [obs_1 … obs_T | q_1 … q_H]  ->  Mamba blocks  ->  head on q_1 … q_H

``q_h`` are learned "future step" query tokens. Because the SSM is causal, the
state built from the observations flows into the queries, and each query
emits the acceleration for its step, and the kinematics (velocity, then
position) is integrated in closed form from the measured state, so the
forecast is smooth and continuous by construction. ``output="delta"`` keeps the
earlier variant (free per-step displacements) for the ablation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import OUTPUT_MODES, integrate_accel, kinematic_features


def selective_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Discretised S6 recurrence (zero-order hold on A, Euler on B).

    x, dt: (b, L, d)   A: (d, n)   B, C: (b, L, n)   D: (d,)

        h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t
        y_t = <C_t, h_t> + D * x_t
    """
    b, _, d = x.shape
    dA = torch.exp(dt.unsqueeze(-1) * A)  # (b, L, d, n)
    dBx = (dt * x).unsqueeze(-1) * B.unsqueeze(2)  # (b, L, d, n)
    h = x.new_zeros(b, d, A.shape[1])
    ys = []
    # unbind (not dA[:, t]) so autograd does one stack instead of L full-size scatters.
    for dA_t, dBx_t, C_t in zip(dA.unbind(1), dBx.unbind(1), C.unbind(1)):
        h = dA_t * h + dBx_t
        ys.append(torch.einsum("bdn,bn->bd", h, C_t))
    return torch.stack(ys, dim=1) + x * D


class CausalDepthwiseConv(nn.Module):
    """Causal depthwise Conv1d written as a sum of shifted copies.

    Numerically identical to ``nn.Conv1d(d, d, k, groups=d)`` with left padding,
    but avoids PyTorch's very slow depthwise-conv backward on CPU (~50x faster here).
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        bound = 1.0 / math.sqrt(kernel_size)
        self.weight = nn.Parameter(torch.empty(kernel_size, channels).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(channels).uniform_(-bound, bound))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (b, L, d)
        k, length = self.weight.shape[0], x.shape[1]
        xp = F.pad(x, (0, 0, k - 1, 0))
        return sum(xp[:, i : i + length] * self.weight[i] for i in range(k)) + self.bias


class MambaBlock(nn.Module):
    """in_proj -> causal depthwise conv -> selective SSM, gated by SiLU(z) -> out_proj."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_range: tuple[float, float] = (1e-3, 1e-1),
    ):
        super().__init__()
        d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = CausalDepthwiseConv(d_inner, d_conv)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # S4D-real initialisation: A = -[1, 2, …, n] for every channel.
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Initialise dt so that softplus(bias) is log-uniform in dt_range.
        lo, hi = math.log(dt_range[0]), math.log(dt_range[1])
        dt = torch.exp(torch.rand(d_inner) * (hi - lo) + lo)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x, z = self.in_proj(u).chunk(2, dim=-1)
        x = F.silu(self.conv(x))

        dt, B, C = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # input-dependent step size: the "selection"
        y = selective_scan(x, dt, -torch.exp(self.A_log), B, C, self.D)
        return self.out_proj(y * F.silu(z))


class ResidualBlock(nn.Module):
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mixer = MambaBlock(d_model, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class MambaForecaster(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_layers: int = 4,
        d_state: int = 16,
        pred_len: int = 30,
        dt_range: tuple[float, float] = (1e-3, 1e-1),
        context_queries: bool = False,
        output: str = "accel",
    ):
        super().__init__()
        assert output in OUTPUT_MODES, output
        self.output = output
        self.pred_len = pred_len
        self.context_queries = context_queries
        self.embed = nn.Linear(4, d_model)
        self.queries = nn.Parameter(torch.randn(pred_len, d_model) * 0.02)
        self.layers = nn.ModuleList(
            ResidualBlock(d_model, d_state=d_state, dt_range=tuple(dt_range)) for _ in range(n_layers)
        )
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, T, 2) relative coords -> (B, H, 2) relative coords."""
        tokens = self.embed(kinematic_features(obs))
        queries = self.queries.unsqueeze(0).expand(obs.shape[0], -1, -1)
        if self.context_queries:
            # Give every future query the embedding of the present state, so the
            # forecast does not depend only on what the scan carries forward.
            queries = queries + tokens[:, -1:]
        x = torch.cat([tokens, queries], dim=1)
        for layer in self.layers:
            x = layer(x)
        out = self.head(self.norm(x[:, -self.pred_len :]))
        if self.output == "accel":
            return integrate_accel(obs, out)
        return obs[:, -1:, :] + torch.cumsum(out, dim=1)
