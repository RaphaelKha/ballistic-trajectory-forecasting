"""Synthetic (observation, future) windows for training and evaluation.

Each sample is a window cut at a random offset of a longer flight so that the
model sees every phase (ascent, apex, descent). Observations are noisy and
pixel-quantised like a real detector; targets are the *true* future positions.

Everything is expressed relative to the last observed point and divided by
``SCALE``: this gives translation invariance and O(1) magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .physics import REGIMES, PhysicsConfig, observe, random_shots

SCALE = 100.0  # px per normalised unit


@dataclass
class WindowConfig:
    obs_len: int = 30
    pred_len: int = 30
    flight_len: int = 100
    noise_px: float = 1.0


def to_relative(obs_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(…, T, 2) absolute px -> normalised coords relative to the last point."""
    anchor = obs_px[..., -1:, :]
    return (obs_px - anchor) / SCALE, anchor


def to_absolute(pred_rel: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    return pred_rel * SCALE + anchor


def make_windows(
    n: int,
    regime: str,
    wcfg: WindowConfig,
    pcfg: PhysicsConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Return absolute-pixel arrays ``obs`` (n, T, 2), ``future`` (n, H, 2), ``obs_clean``."""
    total = wcfg.obs_len + wcfg.pred_len
    flights, params = random_shots(n, regime, wcfg.flight_len, pcfg, rng)
    starts = rng.integers(0, wcfg.flight_len - total + 1, size=n)
    idx = starts[:, None] + np.arange(total)[None, :]
    windows = np.take_along_axis(flights, idx[..., None], axis=1)

    clean_obs = windows[:, : wcfg.obs_len]
    return {
        "obs": observe(clean_obs, wcfg.noise_px, rng),
        "obs_clean": clean_obs,
        "future": windows[:, wcfg.obs_len :],
        "drag": params.drag,
        "wind": params.wind,
    }


def make_mixed_windows(
    n: int, wcfg: WindowConfig, pcfg: PhysicsConfig, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Equal mix of all physical regimes, shuffled."""
    parts = [make_windows(n // len(REGIMES), r, wcfg, pcfg, rng) for r in REGIMES]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    perm = rng.permutation(len(merged["obs"]))
    return {k: v[perm] for k, v in merged.items()}


class TrajectoryDataset(Dataset):
    """Normalised tensors ready for the neural forecasters."""

    def __init__(self, windows: dict[str, np.ndarray]):
        obs_rel, anchor = to_relative(windows["obs"])
        self.obs = torch.as_tensor(obs_rel, dtype=torch.float32)
        self.future = torch.as_tensor((windows["future"] - anchor) / SCALE, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.obs[i], self.future[i]
