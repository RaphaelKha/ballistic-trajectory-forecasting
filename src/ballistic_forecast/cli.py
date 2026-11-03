"""Command-line entry point: ``ballistic <command>``.

ballistic train --config configs/mamba.yaml
ballistic benchmark
ballistic figures
ballistic demo --out assets/demo.mp4 --gif assets/demo.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _train(args: argparse.Namespace) -> None:
    from .train import TrainConfig, train

    cfg = yaml.safe_load(Path(args.config).read_text())
    for key in ("epochs", "samples_per_epoch", "threads"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    train(TrainConfig.from_dict(cfg))


def _benchmark(args: argparse.Namespace) -> None:
    from .benchmark import run_benchmark

    run_benchmark(args.checkpoints, args.out, n_per_regime=args.n)


def _figures(args: argparse.Namespace) -> None:
    from .viz.plots import make_all_figures

    make_all_figures(args.checkpoints, args.results, args.out)


def _demo(args: argparse.Namespace) -> None:
    from .demo import run_demo

    run_demo(
        args.checkpoints,
        out_mp4=args.out,
        out_gif=args.gif,
        regime=args.regime,
        n_shots=args.shots,
        frames=args.frames,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ballistic", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a neural forecaster")
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--samples-per-epoch", dest="samples_per_epoch", type=int)
    p.add_argument("--threads", type=int)
    p.set_defaults(func=_train)

    p = sub.add_parser("benchmark", help="evaluate every model on every physical regime")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--out", default="results")
    p.add_argument("--n", type=int, default=5000, help="test windows per regime")
    p.set_defaults(func=_benchmark)

    p = sub.add_parser("figures", help="render README figures from benchmark results")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="assets")
    p.set_defaults(func=_figures)

    p = sub.add_parser("demo", help="render the radar HUD video (detect -> track -> forecast)")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--out", default="assets/demo.mp4")
    p.add_argument("--gif", default=None)
    p.add_argument("--regime", default="drag_wind")
    p.add_argument("--shots", type=int, default=14)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
