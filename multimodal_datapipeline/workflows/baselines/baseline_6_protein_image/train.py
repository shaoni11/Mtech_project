#!/usr/bin/env python3
"""Train Baseline 6: protein + cell-image activity classification.

Runs the Baseline 7 trainer on the same aligned rows with ``--modalities protein,image``.
All Baseline 7 options are accepted (e.g. ``--protein-encoder structure_gnn``,
``--fusion concat``, ``--split cold_target``).
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = PROJECT_ROOT / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from multimodal_datapipeline.utils.baseline_launcher import load_baseline_module  # noqa: E402


DEFAULTS = {
    "--modalities": "protein,image",
    "--out-dir": str(PROJECT_ROOT / "results" / "baseline_6_protein_image"),
}


def main() -> None:
    for flag, value in DEFAULTS.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:]):
            sys.argv += [flag, value]
    load_baseline_module("baseline_7_molecule_protein_image").main()


if __name__ == "__main__":
    main()
