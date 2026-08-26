"""Project path helpers used by workflows and package entrypoints."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    configured = os.environ.get("MULTIMODAL_DATAPIPELINE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def package_root() -> Path:
    return project_root() / "package"


def data_root() -> Path:
    return project_root() / "data"


def results_root() -> Path:
    return project_root() / "results"


def workflow_root() -> Path:
    return project_root() / "workflows"

