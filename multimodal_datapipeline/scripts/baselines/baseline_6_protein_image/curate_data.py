#!/usr/bin/env python3
"""Check whether protein + image alignment is available."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUMMARY = PROJECT_ROOT / "data" / "processed" / "baseline_6_protein_image_summary.json"


def main() -> None:
    summary = {
        "status": "blocked",
        "reason": "BBBC021 image rows do not currently contain target/protein annotations.",
        "required_alignment": [
            "image row -> compound",
            "compound -> known target",
            "target -> UniProt/protein sequence",
        ],
        "recommended_action": "Treat Baseline 6 as exploratory until compound-target labels are curated.",
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

