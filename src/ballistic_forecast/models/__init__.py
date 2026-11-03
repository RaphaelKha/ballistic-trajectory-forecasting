from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..data import SCALE, to_relative
from .baselines import constant_velocity, parabola_fit
from .gru import GRUForecaster
from .mamba import MambaForecaster

NEURAL_MODELS: dict[str, type[nn.Module]] = {
    "gru": GRUForecaster,
    "mamba": MambaForecaster,
}
BASELINES = {
    "const_velocity": constant_velocity,
    "parabola": parabola_fit,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in NEURAL_MODELS:
        raise KeyError(f"unknown model {name!r}, expected one of {list(NEURAL_MODELS)}")
    return NEURAL_MODELS[name](**kwargs)


def load_checkpoint(path: str | Path, device: str = "cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["name"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict_px(
    model: nn.Module, obs_px: np.ndarray, batch_size: int = 2048, device: str = "cpu"
) -> np.ndarray:
    """Run a neural forecaster on absolute-pixel tracks (N, T, 2) -> (N, H, 2)."""
    model.eval()
    rel, anchor = to_relative(obs_px)
    out = []
    for i in range(0, len(rel), batch_size):
        x = torch.as_tensor(rel[i : i + batch_size], dtype=torch.float32, device=device)
        out.append(model(x).cpu().numpy())
    return np.concatenate(out) * SCALE + anchor


def forecast(name: str, obs_px: np.ndarray, horizon: int, models: dict[str, nn.Module]):
    """Uniform entry point for baselines and neural models (absolute px in/out)."""
    if name in BASELINES:
        return BASELINES[name](obs_px, horizon)
    return predict_px(models[name], obs_px)
