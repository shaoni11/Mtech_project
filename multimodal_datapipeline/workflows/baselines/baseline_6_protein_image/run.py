#!/usr/bin/env python3
"""Run Baseline 6 training entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = PROJECT_ROOT / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from multimodal_datapipeline.utils.baseline_launcher import run_baseline_from_file  # noqa: E402


if __name__ == "__main__":
    run_baseline_from_file(__file__)
