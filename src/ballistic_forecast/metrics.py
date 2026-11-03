"""Standard trajectory-forecasting metrics, all in pixels."""

from __future__ import annotations

import numpy as np


def displacement(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Euclidean error per sample and per future step: (N, H)."""
    return np.linalg.norm(pred - target, axis=-1)


def ade(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Average Displacement Error over the horizon, per sample."""
    return displacement(pred, target).mean(axis=-1)


def fde(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Final Displacement Error (last horizon step), per sample."""
    return displacement(pred, target)[:, -1]


def summarize(pred: np.ndarray, target: np.ndarray, hit_radius: float = 10.0) -> dict[str, float]:
    f = fde(pred, target)
    return {
        "ADE": float(ade(pred, target).mean()),
        "FDE_mean": float(f.mean()),
        "FDE_median": float(np.median(f)),
        "FDE_p90": float(np.percentile(f, 90)),
        f"hit@{hit_radius:g}px": float((f <= hit_radius).mean()),
    }
