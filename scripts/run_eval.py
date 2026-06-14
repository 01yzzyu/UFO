#!/usr/bin/env python3
"""One-command evaluation: infer -> score -> tables.

Prefer:  ufo-eval run ...   or   python -m ufo_bench run ...

Examples:
    python scripts/run_eval.py --run-config configs/run.yaml
    python scripts/run_eval.py --models GPT-5.1 --split mcq --limit 30 --out outputs/demo
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ufo_bench.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["run"] + sys.argv[1:])
