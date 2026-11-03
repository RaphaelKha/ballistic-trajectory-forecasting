from __future__ import annotations

import torch


def kinematic_features(obs: torch.Tensor) -> torch.Tensor:
    """(B, T, 2) positions -> (B, T, 4) [position, 10 * velocity].

    Velocities are ~0.1 in normalised units; the x10 brings them to O(1).
    """
    vel = torch.diff(obs, dim=1, prepend=obs[:, :1])
    return torch.cat([obs, vel * 10.0], dim=-1)


ACC_SCALE = 0.01  # a raw head output of 1.0 means 1 px/frame^2 (gravity is ~0.5)
OUTPUT_MODES = ("accel", "delta")


def initial_velocity(obs: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Velocity at the last observation, averaged over k frames to tame noise: (B, 2)."""
    return (obs[:, -1] - obs[:, -1 - k]) / k


def integrate_accel(obs: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
    """Physics-informed decoding: the network predicts accelerations (B, H, 2) and
    kinematics is hard-coded. The forecast starts at the measured position and
    velocity, so it is continuous and smooth by construction."""
    vel = initial_velocity(obs)[:, None] + torch.cumsum(acc * ACC_SCALE, dim=1)
    return obs[:, -1:] + torch.cumsum(vel, dim=1)
