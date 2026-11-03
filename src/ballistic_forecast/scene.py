"""Multi-projectile scene used by the live demo: staggered salvos fired from
both sides of the screen (so tracks cross), one shared wind, and an occlusion
band the projectiles disappear behind."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .physics import PhysicsConfig, sample_params, simulate


@dataclass
class Scene:
    truth: np.ndarray  # (S, F, 2) true positions, NaN before launch
    launch: np.ndarray  # (S,) launch frame of each shot
    width: int
    height: int
    occluders: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return self.truth.shape[1]

    def visible(self, f: int) -> np.ndarray:
        """(S,) mask of shots that are airborne, on screen and not occluded at frame f."""
        p = self.truth[:, f]
        ok = ~np.isnan(p[:, 0])
        with np.errstate(invalid="ignore"):
            ok &= (p[:, 0] >= 0) & (p[:, 0] < self.width) & (p[:, 1] >= 0) & (p[:, 1] < self.height)
            for x0, y0, x1, y1 in self.occluders:
                ok &= ~((p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1))
        return ok

    def render_sensor(self, f: int, radius: int = 4) -> np.ndarray:
        """Single-channel frame the detector sees (no HUD, no trails)."""
        img = np.zeros((self.height, self.width), dtype=np.uint8)
        for x, y in self.truth[self.visible(f), f]:
            # shift=4 -> sub-pixel centre, like a real blurred blob.
            cv2.circle(img, (int(x * 16), int(y * 16)), radius * 16, 255, -1, cv2.LINE_AA, 4)
        return img


def make_scene(
    n_shots: int = 14,
    frames: int = 240,
    regime: str = "drag_wind",
    seed: int = 7,
    width: int = 1280,
    height: int = 720,
    occlusion: bool = True,
) -> Scene:
    rng = np.random.default_rng(seed)
    params = sample_params(n_shots, regime, PhysicsConfig(), rng)
    params.wind[:] = params.wind[0]  # one atmosphere for the whole scene

    launch = np.sort(rng.integers(0, max(1, frames - 110), size=n_shots))
    from_left = rng.random(n_shots) < 0.5
    x0 = np.where(from_left, rng.uniform(40, 360, n_shots), rng.uniform(width - 360, width - 40, n_shots))
    y0 = height - rng.uniform(5, 40, n_shots)
    vx = np.where(from_left, 1, -1) * rng.uniform(9, 17, n_shots)
    vy = -rng.uniform(19, 26, n_shots)

    flights = simulate(np.stack([x0, y0], -1), np.stack([vx, vy], -1), params, frames)
    truth = np.full((n_shots, frames, 2), np.nan)
    for s in range(n_shots):
        truth[s, launch[s] :] = flights[s, : frames - launch[s]]

    occluders = [(width // 2 - 70, 140, width // 2 + 70, 330)] if occlusion else []
    return Scene(truth, launch, width, height, occluders)
