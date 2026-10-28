#!/usr/bin/env python3
"""Baseline 6 uses the aligned compound-protein-image table built for Baseline 7.

Alignment: BBBC021 image -> compound -> ChEMBL binding activity -> protein (UniProt,
AlphaFold). This script builds that table if it is missing and records a summary.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALIGNED = PROJECT_ROOT / "data" / "processed" / "baseline_7_molecule_protein_image.csv"
B7_CURATE = PROJECT_ROOT / "workflows" / "baselines" / "baseline_7_molecule_protein_image" / "curate_data.py"
SUMMARY = PROJECT_ROOT / "data" / "processed" / "baseline_6_protein_image_summary.json"


def main() -> None:
    if not ALIGNED.exists():
        subprocess.run([sys.executable, str(B7_CURATE)], check=True)
    summary = {
        "status": "ready",
        "aligned_table": str(ALIGNED),
        "modalities": ["protein", "image"],
        "note": "Same rows as Baseline 7; train with baseline_6_protein_image/train.py.",
    }
    with SUMMARY.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
