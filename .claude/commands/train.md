---
description: Train a forecaster from its config (usage: /train mamba)
argument-hint: <model name, e.g. mamba or gru>
---
Train the `$ARGUMENTS` model with `.venv/bin/ballistic train --config configs/$ARGUMENTS.yaml`,
running it in the background and logging to `logs/train_$ARGUMENTS.log`.
When it finishes, report the best validation FDE and the epoch it was reached, then
offer to run /benchmark.
