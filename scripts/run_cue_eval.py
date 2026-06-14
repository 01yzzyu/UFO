#!/usr/bin/env python3
"""Thin wrapper for `ufo-eval cue-eval` (kept for backward compatibility).

Prefer:  ufo-eval cue-eval ...   or   python -m ufo_bench cue-eval ...
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ufo_bench.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["cue-eval"] + sys.argv[1:])
