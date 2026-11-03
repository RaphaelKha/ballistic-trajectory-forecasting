"""Evaluate every forecaster on held-out windows of every physical regime.

The test seed (20_000+) is disjoint from training (0) and validation (10_000).
Results go to ``results/benchmark.json`` and a Markdown table for the README.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import WindowConfig, make_windows
from .metrics import displacement, summarize
from .models import BASELINES, count_parameters, forecast, load_checkpoint
from .physics import REGIMES, PhysicsConfig

MODEL_ORDER = ("const_velocity", "parabola", "gru", "mamba")
PRETTY = {
    "const_velocity": "Constant velocity",
    "parabola": "Parabola fit (physics)",
    "gru": "GRU seq2seq",
    "mamba": "Mamba-SSM",
}


def load_test_sets(n_per_regime: int, wcfg: WindowConfig | None = None) -> dict[str, dict]:
    wcfg = wcfg or WindowConfig()
    return {
        regime: make_windows(n_per_regime, regime, wcfg, PhysicsConfig(), np.random.default_rng(20_000 + i))
        for i, regime in enumerate(REGIMES)
    }


def load_models(checkpoints: str | Path) -> dict:
    ckpt_dir = Path(checkpoints)
    return {p.stem: load_checkpoint(p) for p in sorted(ckpt_dir.glob("*.pt"))}


def run_benchmark(checkpoints: str = "checkpoints", out: str = "results", n_per_regime: int = 5000):
    models = load_models(checkpoints)
    names = [m for m in MODEL_ORDER if m in BASELINES or m in models]
    tests = load_test_sets(n_per_regime)

    results: dict = {"n_per_regime": n_per_regime, "params": {}, "regimes": {}, "fde_by_step": {}}
    for name in names:
        results["params"][name] = count_parameters(models[name]) if name in models else 0

    for regime, data in tests.items():
        results["regimes"][regime] = {}
        results["fde_by_step"][regime] = {}
        for name in names:
            pred = forecast(name, data["obs"], data["future"].shape[1], models)
            results["regimes"][regime][name] = summarize(pred, data["future"])
            results["fde_by_step"][regime][name] = displacement(pred, data["future"]).mean(0).tolist()

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(results, indent=1))
    table = markdown_table(results)
    (out_dir / "benchmark.md").write_text(table)
    print(table)
    if (Path(checkpoints) / "ablations").is_dir():
        run_ablation(checkpoints, out, n_per_regime)
    return results


def markdown_table(results: dict) -> str:
    regimes = list(results["regimes"])
    names = list(results["regimes"][regimes[0]])
    header = "| Model | Params | " + " | ".join(f"{r} FDE (median / mean)" for r in regimes) + " |"
    sep = "|---|---:|" + "---:|" * len(regimes)
    rows = []
    for name in names:
        best = [
            min(results["regimes"][r], key=lambda m: results["regimes"][r][m]["FDE_median"]) for r in regimes
        ]
        cells = []
        for r, b in zip(regimes, best):
            s = results["regimes"][r][name]
            txt = f"{s['FDE_median']:.1f} / {s['FDE_mean']:.1f} px"
            cells.append(f"**{txt}**" if name == b else txt)
        n = results["params"][name]
        rows.append(f"| {PRETTY[name]} | {f'{n / 1e3:.0f}k' if n else '0'} | " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows]) + "\n"


ABLATION_LABELS = {
    "gru_delta": "GRU, per-step displacement output (original design)",
    "mamba_delta": "Mamba, per-step displacement output",
    "mamba_ctx": "Mamba, displacement output + larger Δ init + context queries",
    "gru": "GRU, acceleration output (final)",
    "mamba": "Mamba, acceleration output (final)",
}


def run_ablation(checkpoints: str = "checkpoints", out: str = "results", n_per_regime: int = 5000) -> str:
    """Median FDE per regime for the final models and every checkpoint in ``ablations/``."""
    ckpt_dir = Path(checkpoints)
    models = {**load_models(ckpt_dir / "ablations"), **load_models(ckpt_dir)}
    tests = load_test_sets(n_per_regime)
    order = [k for k in ABLATION_LABELS if k in models] + [k for k in models if k not in ABLATION_LABELS]
    lines = [
        "| Variant | Params | " + " | ".join(f"{r} FDE median" for r in tests) + " |",
        "|---|---:|" + "---:|" * len(tests),
    ]
    for name in order:
        cells = []
        for data in tests.values():
            pred = forecast(name, data["obs"], data["future"].shape[1], models)
            cells.append(f"{summarize(pred, data['future'])['FDE_median']:.1f} px")
        label = ABLATION_LABELS.get(name, name)
        lines.append(f"| {label} | {count_parameters(models[name]) / 1e3:.0f}k | " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "ablation.md").write_text(table)
    print(table)
    return table
