#!/usr/bin/env python3
"""Run Baseline 1: molecule-only activity prediction."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.molecule_only_baseline import main  # noqa: E402


if __name__ == "__main__":
    main()

