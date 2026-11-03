"""Blob detector standing in for a real object detector (e.g. YOLO).

Works on a single-channel "sensor" frame: threshold, connected components,
intensity-weighted centroids (sub-pixel). Two projectiles that overlap merge
into one blob, which is exactly the failure mode the tracker must survive.
"""

from __future__ import annotations

import cv2
import numpy as np


def detect(sensor: np.ndarray, threshold: int = 64, min_area: int = 4) -> np.ndarray:
    """Return (K, 2) float centroids in px."""
    mask = (sensor >= threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, w, h = stats[i, :4]
        roi = sensor[y : y + h, x : x + w].astype(np.float64) * (labels[y : y + h, x : x + w] == i)
        total = roi.sum()
        ys, xs = np.mgrid[y : y + h, x : x + w]
        out.append(((roi * xs).sum() / total, (roi * ys).sum() / total))
    return np.array(out, dtype=np.float64).reshape(-1, 2)
