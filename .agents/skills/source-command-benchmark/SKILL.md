---
name: "source-command-benchmark"
description: "Re-run the benchmark and regenerate every README figure"
---

# source-command-benchmark

Use this skill when the user asks to run the migrated source command `benchmark`.

## Command Template

Run `.venv/bin/ballistic benchmark` then `.venv/bin/ballistic figures`.
Show the new `results/benchmark.md` table, compare it with the table in README.md,
and update the README table and any numbers quoted in its prose if they changed.
Look at each regenerated PNG in `assets/` and report anything that renders badly.
