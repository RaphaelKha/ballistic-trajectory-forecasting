"""Radar-style HUD rendering with OpenCV (BGR colours)."""

from __future__ import annotations

import cv2
import numpy as np

BG = (20, 15, 11)
GRID = (44, 37, 30)
GRID_MAJOR = (60, 50, 40)
TEXT = (236, 236, 236)
MUTED = (150, 145, 140)
TRAIL = (175, 170, 165)
HEAD = (255, 255, 255)

# Same categorical palette as the matplotlib figures (validated for CVD on dark).
MODEL_COLORS = {
    "mamba": (0, 133, 201),  # yellow  #c98500
    "gru": (112, 158, 25),  # aqua    #199e70
    "parabola": (38, 89, 217),  # orange  #d95926
}
MODEL_LABELS = {"mamba": "Mamba-SSM", "gru": "GRU seq2seq", "parabola": "Parabola fit"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def background(width: int, height: int, step: int = 80) -> np.ndarray:
    img = np.full((height, width, 3), BG, dtype=np.uint8)
    for i, x in enumerate(range(0, width, step)):
        cv2.line(img, (x, 0), (x, height), GRID_MAJOR if i % 4 == 0 else GRID, 1)
    for i, y in enumerate(range(height, 0, -step)):
        cv2.line(img, (0, y), (width, y), GRID_MAJOR if i % 4 == 0 else GRID, 1)
    # ground line
    cv2.line(img, (0, height - 2), (width, height - 2), (80, 70, 60), 2)
    return img


def draw_occluder(img: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (70, 62, 55), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
    for d in range(-(y1 - y0), x1 - x0, 14):
        p0 = (x0 + max(d, 0), y0 + max(-d, 0))
        p1 = (x0 + min(d + (y1 - y0), x1 - x0), y0 + min(x1 - x0 - d, y1 - y0))
        cv2.line(img, p0, p1, (95, 85, 75), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), (120, 110, 100), 1)
    cv2.putText(img, "OCCLUSION", (x0 + 8, y0 + 18), FONT, 0.42, MUTED, 1, cv2.LINE_AA)


def polyline(img, pts: np.ndarray, color, thickness: int = 2, dashed: bool = False) -> None:
    pts = np.asarray(pts)
    if dashed:
        for i in range(0, len(pts) - 1, 2):
            cv2.line(img, _pt(pts[i]), _pt(pts[i + 1]), color, thickness, cv2.LINE_AA)
    else:
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_track(img, track_id: int, history: np.ndarray, coasting: bool) -> None:
    n = len(history)
    for i, p in enumerate(history[:-1]):
        a = (i + 1) / n
        color = tuple(int(c * a + b * (1 - a)) for c, b in zip(TRAIL, BG))
        cv2.circle(img, _pt(p), 2, color, -1, cv2.LINE_AA)
    head = history[-1]
    if coasting:
        cv2.circle(img, _pt(head), 9, MUTED, 1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"T{track_id:02d} COAST",
            (int(head[0]) + 11, int(head[1]) - 8),
            FONT,
            0.4,
            MUTED,
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.circle(img, _pt(head), 4, HEAD, -1, cv2.LINE_AA)
        cv2.circle(img, _pt(head), 9, TRAIL, 1, cv2.LINE_AA)
        cv2.putText(
            img, f"T{track_id:02d}", (int(head[0]) + 11, int(head[1]) - 8), FONT, 0.42, TEXT, 1, cv2.LINE_AA
        )


def draw_forecast(img, pred: np.ndarray, model: str, primary: bool) -> None:
    color = MODEL_COLORS[model]
    polyline(img, pred, color, 2 if primary else 1, dashed=not primary)
    if primary:
        end = _pt(pred[-1])
        cv2.drawMarker(img, end, color, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        cv2.circle(img, end, 7, color, 1, cv2.LINE_AA)


def draw_panel(img, frame: int, fps: int, n_tracks: int, live_fde: dict[str, list[float]]) -> None:
    x, y, w, h = 16, 16, 330, 150
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (10, 8, 6), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, dst=img)
    cv2.rectangle(img, (x, y), (x + w, y + h), GRID_MAJOR, 1)

    cv2.putText(img, "BALLISTIC FORECAST // +1.0 s", (x + 12, y + 24), FONT, 0.52, TEXT, 1, cv2.LINE_AA)
    cv2.putText(
        img,
        f"T+{frame / fps:5.2f}s   tracks {n_tracks:2d}",
        (x + 12, y + 46),
        FONT,
        0.42,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(img, "live median FDE", (x + 200, y + 46), FONT, 0.38, MUTED, 1, cv2.LINE_AA)
    for i, model in enumerate(("mamba", "gru", "parabola")):
        yy = y + 74 + i * 24
        color = MODEL_COLORS[model]
        cv2.line(img, (x + 12, yy - 5), (x + 34, yy - 5), color, 2, cv2.LINE_AA)
        cv2.putText(img, MODEL_LABELS[model], (x + 44, yy), FONT, 0.45, TEXT, 1, cv2.LINE_AA)
        errs = live_fde.get(model, [])
        val = f"{np.median(errs):6.1f} px" if errs else "   --   "
        cv2.putText(img, val, (x + 210, yy), FONT, 0.48, color, 1, cv2.LINE_AA)


def draw_legend(img) -> None:
    h, w = img.shape[:2]
    y = h - 18
    items = [
        ("track", TRAIL),
        ("Mamba forecast", MODEL_COLORS["mamba"]),
        ("parabola baseline", MODEL_COLORS["parabola"]),
    ]
    x = w - 470
    for label, color in items:
        cv2.circle(img, (x, y - 4), 4, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x + 10, y), FONT, 0.42, MUTED, 1, cv2.LINE_AA)
        x += 20 + cv2.getTextSize(label, FONT, 0.42, 1)[0][0] + 14
