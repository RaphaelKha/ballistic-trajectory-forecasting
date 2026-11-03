"""Training loop shared by every neural forecaster.

Fresh synthetic data is generated every epoch (the simulator is cheap), so the
models never see the same window twice and cannot overfit the training set.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SCALE, TrajectoryDataset, WindowConfig, make_mixed_windows
from .metrics import fde
from .models import build_model, count_parameters
from .physics import PhysicsConfig


@dataclass
class TrainConfig:
    model: str = "mamba"
    run_name: str = ""  # checkpoint name, defaults to the model name
    model_kwargs: dict = field(default_factory=dict)
    epochs: int = 30
    samples_per_epoch: int = 48_000
    val_samples: int = 6_000
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 0
    threads: int = 4
    out_dir: str = "checkpoints"
    window: WindowConfig = field(default_factory=WindowConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        d = dict(d)
        window = WindowConfig(**d.pop("window", {}))
        physics = PhysicsConfig(**d.pop("physics", {}))
        return cls(window=window, physics=physics, **d)


def horizon_loss(pred: torch.Tensor, target: torch.Tensor, terminal_weight: float = 2.0):
    """Euclidean error in px, weighted up along the horizon + an extra terminal term.

    A simplified version of the original "sniper loss": far-future steps matter
    more, and the final point (the FDE) gets its own penalty.
    """
    err = torch.sqrt(((pred - target) * SCALE).pow(2).sum(-1) + 1e-6)  # (B, H)
    weights = torch.linspace(1.0, 3.0, err.shape[1], device=err.device)
    return (err * weights).mean() / weights.mean() + terminal_weight * err[:, -1].mean()


@torch.no_grad()
def evaluate(model, loader) -> dict[str, float]:
    model.eval()
    preds, targets = [], []
    for obs, fut in loader:
        preds.append(model(obs).numpy())
        targets.append(fut.numpy())
    p, t = np.concatenate(preds) * SCALE, np.concatenate(targets) * SCALE
    f = fde(p, t)
    return {"val_ADE": float(np.linalg.norm(p - t, axis=-1).mean()), "val_FDE": float(f.mean())}


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)

    run = cfg.run_name or cfg.model
    model = build_model(cfg.model, pred_len=cfg.window.pred_len, **cfg.model_kwargs)
    n_params = count_parameters(model)
    print(f"[{run}] {n_params:,} parameters")

    val_set = TrajectoryDataset(
        make_mixed_windows(cfg.val_samples, cfg.window, cfg.physics, np.random.default_rng(10_000))
    )
    val_loader = DataLoader(val_set, batch_size=1024)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = cfg.samples_per_epoch // cfg.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.1
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{run}.pt"
    history, best = [], float("inf")
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_set = TrajectoryDataset(make_mixed_windows(cfg.samples_per_epoch, cfg.window, cfg.physics, rng))
        loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        running = 0.0
        for obs, fut in loader:
            loss = horizon_loss(model(obs), fut)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()

        metrics = evaluate(model, val_loader)
        row = {
            "epoch": epoch,
            "train_loss": running / len(loader),
            **metrics,
            "lr": sched.get_last_lr()[0],
            "elapsed_s": time.time() - t0,
        }
        history.append(row)
        flag = ""
        if metrics["val_FDE"] < best:
            best = metrics["val_FDE"]
            flag = "  *best*"
            torch.save(
                {
                    "name": cfg.model,
                    "model_kwargs": {"pred_len": cfg.window.pred_len, **cfg.model_kwargs},
                    "state_dict": model.state_dict(),
                    "n_params": n_params,
                    "epoch": epoch,
                    "config": asdict(cfg),
                },
                ckpt_path,
            )
        print(
            f"[{run}] epoch {epoch:3d}/{cfg.epochs}  loss {row['train_loss']:7.3f}  "
            f"val ADE {metrics['val_ADE']:6.2f}px  FDE {metrics['val_FDE']:6.2f}px  "
            f"({row['elapsed_s']:.0f}s){flag}",
            flush=True,
        )
        (out_dir / f"{run}_history.json").write_text(json.dumps(history, indent=1))

    print(f"[{run}] best val FDE {best:.2f}px -> {ckpt_path}")
    return ckpt_path
