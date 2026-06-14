#!/usr/bin/env bash
# End-to-end demo: inference -> scoring -> tables, in ONE command.
#
# Prerequisites:
#   pip install -e .
#   cp .env.example .env   # set OPENROUTER_API_KEY  (or: export OPENROUTER_API_KEY=...)
set -euo pipefail
cd "$(dirname "$0")/.."

# (Optional) pre-download the dataset to data_cache/ for offline/faster runs:
#   ufo-eval download
# Otherwise it is fetched automatically on first use.

# Option A — one command (recommended):
ufo-eval run --models GPT-5.1 Qwen3-VL-8B \
  --split mcq --protocols direct textual visual joint \
  --limit 30 --out outputs/demo_mcq

# Option B — config-driven (edit configs/run.yaml, then):
#   ufo-eval run --run-config configs/run.yaml

echo "Done. See outputs/demo_mcq/tables/summary.md (and results.csv, main_table.tex)"
