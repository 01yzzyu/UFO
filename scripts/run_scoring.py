#!/usr/bin/env python3
"""Thin wrapper for `ufo-eval score` (kept for backward compatibility).

Prefer:  ufo-eval score ...   or   python -m ufo_bench score ...
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ufo_bench.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["score"] + sys.argv[1:])
