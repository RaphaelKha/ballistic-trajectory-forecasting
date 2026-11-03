"""Non-learned physics baselines. They operate on absolute pixel arrays.

If a neural forecaster cannot beat these, it is not worth its parameters.
"""

from __future__ import annotations

import numpy as np


def constant_velocity(obs: np.ndarray, horizon: int, fit_last: int = 5) -> np.ndarray:
    """Extrapolate the mean velocity of the last ``fit_last`` frames."""
    vel = (obs[:, -1] - obs[:, -fit_last]) / (fit_last - 1)
    steps = np.arange(1, horizon + 1)[None, :, None]
    return obs[:, -1:, :] + steps * vel[:, None, :]


def parabola_fit(obs: np.ndarray, horizon: int) -> np.ndarray:
    """Least-squares quadratic fit of x(t) and y(t), then extrapolate.

    This is the exact model of drag-free ballistic flight, i.e. the optimal
    estimator in the ``vacuum`` regime up to measurement noise.
    """
    n, t_obs, _ = obs.shape
    t = np.arange(t_obs, dtype=np.float64)
    t_fut = np.arange(t_obs, t_obs + horizon, dtype=np.float64)
    vander = np.stack([np.ones_like(t), t, t**2], axis=1)  # (T, 3)
    vander_fut = np.stack([np.ones_like(t_fut), t_fut, t_fut**2], axis=1)
    # Solve all 2N least-squares problems at once.
    rhs = obs.transpose(1, 0, 2).reshape(t_obs, n * 2)
    coef, *_ = np.linalg.lstsq(vander, rhs, rcond=None)
    pred = vander_fut @ coef  # (H, 2N)
    return pred.reshape(horizon, n, 2).transpose(1, 0, 2)
