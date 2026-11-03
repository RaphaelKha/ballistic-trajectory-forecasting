"""Vectorised 2D projectile simulator.

Units are pixels and frames (30 fps): positions in px, velocities in px/frame,
accelerations in px/frame^2. Image convention: +y points down, so gravity is +g.

Equation of motion (semi-implicit Euler, dt = 1 frame):

    v_rel = v - w                      # velocity relative to the wind
    a     = [0, g] - k * |v_rel| * v_rel   # gravity + quadratic air drag
    v    <- v + a
    p    <- p + v

Three regimes make the benchmark honest:

* ``vacuum``    k = 0, w = 0  -> exact parabola; a polynomial fit is optimal here.
* ``drag``      k > 0, w = 0  -> non-parabolic, decelerating trajectories.
* ``drag_wind`` k > 0, w != 0 -> drag towards a per-shot unknown wind.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REGIMES = ("vacuum", "drag", "drag_wind")


@dataclass
class PhysicsConfig:
    gravity: tuple[float, float] = (0.35, 0.65)  # px/frame^2, sampled per shot
    drag: tuple[float, float] = (0.0003, 0.0015)  # quadratic drag coefficient k
    wind: tuple[float, float] = (-12.0, 12.0)  # horizontal wind speed, px/frame
    speed_x: tuple[float, float] = (-22.0, 22.0)
    speed_y: tuple[float, float] = (-32.0, 4.0)
    width: int = 1920
    height: int = 1080


@dataclass
class ShotParams:
    """Per-trajectory physical parameters, each an array of shape (N,)."""

    gravity: np.ndarray
    drag: np.ndarray
    wind: np.ndarray


def sample_params(n: int, regime: str, cfg: PhysicsConfig, rng: np.random.Generator) -> ShotParams:
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}, expected one of {REGIMES}")
    gravity = rng.uniform(*cfg.gravity, size=n)
    drag = rng.uniform(*cfg.drag, size=n) if regime != "vacuum" else np.zeros(n)
    wind = rng.uniform(*cfg.wind, size=n) if regime == "drag_wind" else np.zeros(n)
    return ShotParams(gravity, drag, wind)


def simulate(pos0: np.ndarray, vel0: np.ndarray, params: ShotParams, steps: int) -> np.ndarray:
    """Integrate N trajectories for ``steps`` frames. Returns (N, steps, 2)."""
    pos = np.asarray(pos0, dtype=np.float64).copy()
    vel = np.asarray(vel0, dtype=np.float64).copy()
    g = np.stack([np.zeros_like(params.gravity), params.gravity], axis=-1)
    wind = np.stack([params.wind, np.zeros_like(params.wind)], axis=-1)
    k = params.drag[:, None]

    out = np.empty((pos.shape[0], steps, 2), dtype=np.float64)
    for t in range(steps):
        v_rel = vel - wind
        speed = np.linalg.norm(v_rel, axis=-1, keepdims=True)
        vel = vel + g - k * speed * v_rel
        pos = pos + vel
        out[:, t] = pos
    return out


def random_shots(
    n: int, regime: str, steps: int, cfg: PhysicsConfig, rng: np.random.Generator
) -> tuple[np.ndarray, ShotParams]:
    """Sample N random shots anywhere on the canvas and simulate them."""
    pos0 = np.stack([rng.uniform(0, cfg.width, n), rng.uniform(0, cfg.height, n)], axis=-1)
    vel0 = np.stack([rng.uniform(*cfg.speed_x, n), rng.uniform(*cfg.speed_y, n)], axis=-1)
    params = sample_params(n, regime, cfg, rng)
    return simulate(pos0, vel0, params, steps), params


def observe(traj: np.ndarray, noise_px: float, rng: np.random.Generator, quantize: bool = True) -> np.ndarray:
    """Turn clean trajectories into sensor measurements (pixel grid + jitter)."""
    obs = np.round(traj) if quantize else traj.copy()
    if noise_px > 0:
        obs = obs + rng.normal(0.0, noise_px, size=obs.shape)
    return obs
