#!/usr/bin/env python3
"""Run Baseline 1: molecule-only activity prediction."""

from __future__ import annotations

import sys
from pathlib import Path


BASELINE_DIR = Path(__file__).resolve().parent
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from train import main  # noqa: E402


if __name__ == "__main__":
    main()
