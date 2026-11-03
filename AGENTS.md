# AGENTS.md

Guidance for Codex (and humans) working in this repository.

## What this project is

Real-time multi-target ballistic trajectory forecasting on synthetic video:
render → detect → track (Kalman + Hungarian) → forecast 1 s ahead with a
selective state-space model (Mamba), benchmarked against a GRU and physics
baselines. Everything runs on CPU; no Colab, no Google Drive.

## Commands

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"   # setup
.venv/bin/pytest                                      # tests (~2 s)
.venv/bin/ruff check src tests                        # lint

.venv/bin/ballistic train --config configs/mamba.yaml  # ~45 min CPU (gru: ~12 min)
.venv/bin/ballistic benchmark                          # -> results/benchmark.{json,md}
.venv/bin/ballistic figures                            # -> assets/*.png
.venv/bin/ballistic demo --gif assets/demo.gif         # -> assets/demo.mp4 + gif (needs ffmpeg)
```

Quick smoke run: `ballistic train --config configs/gru.yaml --epochs 2 --samples-per-epoch 2560`.

## Layout

```
src/ballistic_forecast/
  physics.py         vectorised simulator: gravity, quadratic drag, wind (px, frames)
  data.py            obs/future windows, relative-coordinate normalisation (SCALE=100 px)
  metrics.py         ADE / FDE / hit-rate, all in px
  models/
    mamba.py         pure-PyTorch Mamba block (S6 selective scan) + forecaster
    gru.py           seq2seq GRU (the original model of the project)
    baselines.py     constant velocity, parabola least-squares fit
    __init__.py      registry, checkpoint loading, predict_px (absolute px in/out)
  train.py           training loop, fresh data every epoch, best-on-val checkpoint
  benchmark.py       per-regime evaluation on a disjoint test seed
  scene.py           multi-shot demo scene (crossings, shared wind, occlusion band)
  tracking/          detector (blob centroids), Kalman CA filter, Hungarian tracker
  viz/hud.py         OpenCV radar HUD; viz/plots.py matplotlib README figures
  demo.py            end-to-end video, scored against simulator ground truth
configs/             one YAML per trained model
notebooks/           private prototypes, git-ignored: never publish, never import
```

## Conventions and invariants

- **Units**: pixels and frames (30 fps). Image coordinates, **+y is down**, gravity is +y.
- **Model I/O**: neural models take `(B, T, 2)` positions *relative to the last observed
  point* divided by `SCALE`, and return `(B, H, 2)` in the same frame. Use
  `models.predict_px` to work in absolute pixels; never feed raw pixels to a model.
- **Seeds**: train = 0, validation = 10_000, test = 20_000 + regime index. Keep them disjoint.
- **Honest evaluation**: score against simulator ground truth, never against tracker output;
  do not clip axes or drop outliers to make numbers look better. Report median *and* mean.
- **Colours** are tied to models (`viz/plots.py:COLORS`, `viz/hud.py:MODEL_COLORS`) and must
  stay identical across figures and video. The palette is CVD-validated; don't add hues ad hoc.
- Checkpoints store `name` + `model_kwargs`; `load_checkpoint` rebuilds the model from them.
  If you change a model's constructor, retrain or keep backwards-compatible kwargs.
- The Mamba scan is a Python loop over time on purpose (portable, L=60). Use `unbind` rather
  than indexing inside the loop — indexing makes the backward quadratic in L.

## Adding a model

1. Implement `nn.Module` with `forward(obs_rel) -> pred_rel` in `models/`.
2. Register it in `models/__init__.py:NEURAL_MODELS`, add a colour + label in the viz modules
   and `benchmark.py:MODEL_ORDER/PRETTY`.
3. Add `configs/<name>.yaml`, a shape test in `tests/test_models.py`, train, re-run
   `benchmark` and `figures`, update the README table from `results/benchmark.md`.
