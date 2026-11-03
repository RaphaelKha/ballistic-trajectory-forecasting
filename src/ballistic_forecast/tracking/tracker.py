"""SORT-style multi-target tracker: Kalman prediction + Hungarian assignment.

Per frame:
  1. every track predicts its next position with its Kalman filter;
  2. the cost matrix (predicted track position <-> detection distance) is solved
     optimally with the Hungarian algorithm, pairs beyond ``gate_px`` rejected;
  3. matched tracks are corrected, unmatched ones *coast* on their prediction
     (this is what carries identities through crossings and occlusions);
  4. unmatched detections spawn tentative tracks, stale tracks are dropped.

Each track keeps a rolling history of positions, the input of the forecasters.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import KalmanCA


@dataclass
class Track:
    id: int
    kf: KalmanCA
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    hits: int = 1
    misses: int = 0
    age: int = 0

    @property
    def position(self) -> np.ndarray:
        return self.kf.position


class MultiTargetTracker:
    def __init__(
        self,
        gate_px: float = 40.0,
        max_misses: int = 12,
        min_hits: int = 3,
        history_len: int = 30,
    ):
        self.gate_px = gate_px
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.history_len = history_len
        self.tracks: list[Track] = []
        self._next_id = 0

    def _spawn(self, xy: np.ndarray) -> None:
        t = Track(self._next_id, KalmanCA(xy), deque(maxlen=self.history_len))
        t.history.append(xy.copy())
        self.tracks.append(t)
        self._next_id += 1

    def update(self, detections: np.ndarray) -> list[Track]:
        predicted = np.array([t.kf.predict() for t in self.tracks]).reshape(-1, 2)
        for t in self.tracks:
            t.age += 1

        matched_tracks, matched_dets = set(), set()
        if len(self.tracks) and len(detections):
            cost = np.linalg.norm(predicted[:, None, :] - detections[None, :, :], axis=-1)
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] <= self.gate_px:
                    matched_tracks.add(r)
                    matched_dets.add(c)
                    track = self.tracks[r]
                    track.kf.update(detections[c])
                    track.history.append(detections[c].copy())
                    track.hits += 1
                    track.misses = 0

        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.misses += 1
                track.history.append(track.position)  # coast on the Kalman prediction

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        for j in range(len(detections)):
            if j not in matched_dets:
                self._spawn(detections[j])
        return self.confirmed()

    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.hits >= self.min_hits and t.misses == 0]

    def coasting(self) -> list[Track]:
        return [t for t in self.tracks if t.hits >= self.min_hits and t.misses > 0]
