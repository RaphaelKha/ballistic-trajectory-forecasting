"""End-to-end live demo: render -> detect -> track -> forecast -> HUD video.

Every forecast is scored against the simulator's ground truth 30 frames later
(not against the tracker), so the live FDE in the HUD is a real error. The
ground truth keeps existing off-screen, so shots leaving the frame are scored
too instead of being silently dropped.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .models import load_checkpoint, parabola_fit, predict_px
from .scene import make_scene
from .tracking.detector import detect
from .tracking.tracker import MultiTargetTracker
from .viz import hud

FPS = 30
HORIZON = 30


def _ffmpeg_writer(path: Path, width: int, height: int) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(FPS),
        "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path),
    ]  # fmt: skip
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _to_gif(mp4: Path, gif: Path, width: int = 900, fps: int = 15) -> None:
    filt = (
        f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    )
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf", filt, str(gif)], check=True)


def run_demo(
    checkpoints: str = "checkpoints",
    out_mp4: str = "assets/demo.mp4",
    out_gif: str | None = None,
    regime: str = "drag_wind",
    n_shots: int = 14,
    frames: int = 240,
    seed: int = 7,
) -> dict:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for the demo video (brew install ffmpeg)")

    ckpt_dir = Path(checkpoints)
    models = {name: load_checkpoint(ckpt_dir / f"{name}.pt") for name in ("mamba", "gru")}
    scene = make_scene(n_shots, frames, regime, seed)
    tracker = MultiTargetTracker()
    base = hud.background(scene.width, scene.height)

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = _ffmpeg_writer(out_mp4, scene.width, scene.height)

    pending: list[tuple[int, int, dict[str, np.ndarray]]] = []  # (shot, due frame, endpoints)
    live_fde: dict[str, list[float]] = {"mamba": [], "gru": [], "parabola": []}

    for f in range(frames):
        tracker.update(detect(scene.render_sensor(f)))
        tracks = [t for t in tracker.tracks if t.hits >= tracker.min_hits]
        ready = [t for t in tracks if len(t.history) == HORIZON]

        preds: dict[str, np.ndarray] = {}
        if ready:
            obs = np.stack([np.array(t.history) for t in ready])
            preds = {name: predict_px(m, obs) for name, m in models.items()}
            preds["parabola"] = parabola_fit(obs, HORIZON)

            # Associate each forecasting track with the true shot it is following.
            truth_now = scene.truth[:, f]
            for i, t in enumerate(ready):
                d = np.linalg.norm(truth_now - t.history[-1], axis=-1)
                d = np.where(np.isnan(d), np.inf, d)
                s = int(np.argmin(d))
                if d[s] < 6.0 and t.misses == 0 and f + HORIZON < frames:
                    pending.append((s, f + HORIZON, {k: v[i, -1] for k, v in preds.items()}))

        still = []
        for shot, due, ends in pending:
            if due == f:
                for name, end in ends.items():
                    live_fde[name].append(float(np.linalg.norm(end - scene.truth[shot, f])))
            else:
                still.append((shot, due, ends))
        pending = still

        img = base.copy()
        for box in scene.occluders:
            hud.draw_occluder(img, box)
        for i in range(len(ready)):
            hud.draw_forecast(img, preds["parabola"][i], "parabola", primary=False)
            hud.draw_forecast(img, preds["mamba"][i], "mamba", primary=True)
        for t in tracks:
            hud.draw_track(img, t.id, np.array(t.history), coasting=t.misses > 0)
        hud.draw_panel(img, f, FPS, len(tracks), live_fde)
        hud.draw_legend(img)
        writer.stdin.write(img.tobytes())

    writer.stdin.close()
    writer.wait()
    if out_gif:
        _to_gif(out_mp4, Path(out_gif))

    summary = {
        name: {"n": len(v), "FDE_median": float(np.median(v)), "FDE_mean": float(np.mean(v))}
        for name, v in live_fde.items()
        if v
    }
    summary["tracks_created"] = tracker._next_id
    summary["shots"] = n_shots
    print(json.dumps(summary, indent=1))
    (out_mp4.parent / f"{out_mp4.stem}_stats.json").write_text(json.dumps(summary, indent=1))
    return summary
