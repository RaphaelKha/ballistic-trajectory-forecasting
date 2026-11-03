"""README figures (matplotlib, dark theme).

Colour follows the model, never its rank: the same four slots are used in every
figure and in the HUD video. Palette validated for colour-vision deficiency on
the dark surface (adjacent CVD dE >= 8.4); every series is also direct-labelled.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..benchmark import MODEL_ORDER, PRETTY, load_models, load_test_sets  # noqa: E402
from ..models import forecast  # noqa: E402

SURFACE = "#1a1a19"
TEXT = "#f0efec"
TEXT_2 = "#c3c2b7"
MUTED = "#8a8983"
GRID = "#2f2f2c"
COLORS = {
    "const_velocity": "#3987e5",
    "parabola": "#d95926",
    "gru": "#199e70",
    "mamba": "#c98500",
}
REGIME_TITLES = {
    "vacuum": "Vacuum  (gravity only)",
    "drag": "Air drag",
    "drag_wind": "Air drag + unknown wind",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "font.size": 9.5,
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
            "legend.labelcolor": TEXT_2,
        }
    )


def _suptitle(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.975, title, fontsize=14, fontweight="bold", color=TEXT, va="top")
    fig.text(0.012, 0.915, subtitle, fontsize=9.5, color=TEXT_2, va="top")


def plot_benchmark(results: dict, out: Path) -> None:
    regimes = list(results["regimes"])
    names = [m for m in MODEL_ORDER if m in results["regimes"][regimes[0]]]
    fig, axes = plt.subplots(1, len(regimes), figsize=(12, 3.6), sharey=True)
    for ax, regime in zip(axes, regimes):
        vals = [results["regimes"][regime][m]["FDE_median"] for m in names]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, height=0.62, color=[COLORS[m] for m in names], edgecolor=SURFACE, linewidth=1.5)
        for yi, v in zip(y, vals):
            ax.text(v * 1.12, yi, f"{v:.1f} px", va="center", fontsize=9, color=TEXT)
        ax.set_xscale("log")
        ax.set_xlim(min(vals) / 2.5, max(vals) * 4)
        ax.set_title(REGIME_TITLES[regime], pad=8)
        ax.set_yticks(y, [PRETTY[m] for m in names])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("median FDE @ +1 s (px, log scale)")
    _suptitle(
        fig,
        "Where does learning pay off?",
        "Median final displacement error 30 frames (1 s) ahead, "
        "5,000 held-out tracks per regime. Lower is better.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_error_vs_horizon(results: dict, out: Path) -> None:
    regimes = list(results["fde_by_step"])
    names = [m for m in MODEL_ORDER if m in results["fde_by_step"][regimes[0]]]
    fig, axes = plt.subplots(1, len(regimes), figsize=(12, 3.6), sharey=True)
    for ax, regime in zip(axes, regimes):
        ends = []
        for m in names:
            err = np.asarray(results["fde_by_step"][regime][m])
            steps = np.arange(1, len(err) + 1) / 30.0
            ax.plot(steps, err, color=COLORS[m], lw=1.8, label=PRETTY[m], solid_capstyle="round")
            ends.append((err[-1], m))
        # direct labels at the line ends, pushed apart in log space so they never collide
        gap, last = 0.14, -np.inf
        for val, m in sorted(ends):
            ly = max(np.log10(val), last + gap)
            last = ly
            ax.annotate(
                PRETTY[m].split(" (")[0],
                (1.0, val),
                xytext=(1.03, 10**ly),
                textcoords="data",
                fontsize=8,
                color=TEXT_2,
                va="center",
            )
        ax.set_yscale("log")
        ax.set_xlim(0, 1.32)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(REGIME_TITLES[regime], pad=8)
        ax.set_xlabel("forecast horizon (s)")
    axes[0].set_ylabel("mean displacement error (px)")
    axes[-1].legend(loc="lower right", fontsize=8)
    _suptitle(fig, "Error growth along the horizon", "Mean Euclidean error at each future step (30 fps).")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_examples(models: dict, out: Path, n: int = 6, seed: int = 3) -> None:
    data = load_test_sets(400)["drag_wind"]
    rng = np.random.default_rng(seed)
    # pick visibly curved, long flights so the differences are legible
    # pick strongly curved flights (truth far from its own chord) so differences are legible
    fut_all = data["future"]
    chord = fut_all[:, -1] - fut_all[:, 0]
    mid = fut_all[:, fut_all.shape[1] // 2] - fut_all[:, 0]
    bend = np.abs(np.cross(chord, mid)) / (np.linalg.norm(chord, axis=-1) + 1e-6)
    idx = rng.choice(np.argsort(bend)[-150:], size=n, replace=False)
    obs, fut = data["obs"][idx], data["future"][idx]
    preds = {m: forecast(m, obs, fut.shape[1], models) for m in ("parabola", "gru", "mamba")}

    fig, axes = plt.subplots(2, n // 2, figsize=(12, 6.4))
    for k, ax in enumerate(axes.flat):
        ax.plot(*obs[k][-12:].T, "o", ms=3, color=MUTED, label="observed (noisy)")
        for m in ("parabola", "gru", "mamba"):
            ax.plot(*preds[m][k].T, color=COLORS[m], lw=1.8, label=PRETTY[m], solid_capstyle="round")
            ax.plot(*preds[m][k][-1], "o", ms=6, color=COLORS[m], mec=SURFACE, mew=1.2)
        ax.plot(*fut[k].T, color=TEXT, lw=1.3, ls=(0, (2, 2)), label="ground truth", zorder=5)
        ax.plot(*fut[k][-1], "x", ms=7, mew=1.6, color=TEXT, zorder=6)
        errs = {m: np.linalg.norm(preds[m][k][-1] - fut[k][-1]) for m in preds}
        ax.set_title(
            f"FDE  Mamba {errs['mamba']:.1f}  ·  GRU {errs['gru']:.1f}  ·  "
            f"parabola {errs['parabola']:.0f} px",
            fontsize=8.5,
            fontweight="normal",
            color=TEXT_2,
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.12)
        ax.invert_yaxis()
        ax.tick_params(labelsize=7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9)
    _suptitle(
        fig,
        "Forecasts under drag + wind",
        "Image coordinates in px (y down). Last 12 observed frames shown, 30 frames (1 s) predicted; "
        "x = true position at +1 s.",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_training(checkpoints: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m in ("gru", "mamba"):
        path = checkpoints / f"{m}_history.json"
        if not path.exists():
            continue
        hist = json.loads(path.read_text())
        ep = [h["epoch"] for h in hist]
        fde = [h["val_FDE"] for h in hist]
        ax.plot(ep, fde, color=COLORS[m], lw=1.8, label=PRETTY[m])
        ax.annotate(
            f"{PRETTY[m].split(' (')[0]}  {min(fde):.1f} px",
            (ep[-1], fde[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=TEXT_2,
            va="center",
        )
    ax.set_yscale("log")
    ax.set_xlabel("epoch (48k fresh synthetic tracks each)")
    ax.set_ylabel("validation mean FDE (px)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.margins(x=0.02)
    _suptitle(fig, "Training", "Mixed regimes, CPU only, same budget for both models.")
    fig.tight_layout(rect=(0, 0, 0.86, 0.84))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def make_all_figures(checkpoints: str = "checkpoints", results_dir: str = "results", out: str = "assets"):
    _style()
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads((Path(results_dir) / "benchmark.json").read_text())
    plot_benchmark(results, out_dir / "benchmark.png")
    plot_error_vs_horizon(results, out_dir / "error_vs_horizon.png")
    plot_examples(load_models(checkpoints), out_dir / "examples.png")
    plot_training(Path(checkpoints), out_dir / "training.png")
    print(f"figures written to {out_dir}/")
