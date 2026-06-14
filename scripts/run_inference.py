#!/usr/bin/env python3
"""Thin wrapper for `ufo-eval infer` (kept for backward compatibility).

Prefer:  ufo-eval infer ...   (after `pip install -e .`)
     or  python -m ufo_bench infer ...
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ufo_bench.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["infer"] + sys.argv[1:])
