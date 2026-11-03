"""Constant-acceleration Kalman filter, state = [x, y, vx, vy, ax, ay]."""

from __future__ import annotations

import numpy as np

_F = np.array(
    [
        [1, 0, 1, 0, 0.5, 0],
        [0, 1, 0, 1, 0, 0.5],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ],
    dtype=np.float64,
)
_H = np.zeros((2, 6))
_H[0, 0] = _H[1, 1] = 1.0


class KalmanCA:
    def __init__(
        self,
        xy: np.ndarray,
        meas_std: float = 1.5,
        jerk_std: float = 0.05,
        init_vel_std: float = 25.0,
        init_acc_std: float = 1.0,
    ):
        self.x = np.zeros(6)
        self.x[:2] = xy
        self.P = np.diag([meas_std**2] * 2 + [init_vel_std**2] * 2 + [init_acc_std**2] * 2)
        self.R = np.eye(2) * meas_std**2
        # Discrete white-jerk process noise, dt = 1.
        q = np.array([[1 / 20, 1 / 8, 1 / 6], [1 / 8, 1 / 3, 1 / 2], [1 / 6, 1 / 2, 1.0]])
        self.Q = np.kron(q, np.eye(2)) * jerk_std**2

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    def predict(self) -> np.ndarray:
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + self.Q
        return self.position

    def update(self, z: np.ndarray) -> None:
        y = z - _H @ self.x
        S = _H @ self.P @ _H.T + self.R
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ _H) @ self.P
