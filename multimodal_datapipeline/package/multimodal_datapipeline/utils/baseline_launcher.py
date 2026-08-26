"""Console entrypoints for baseline workflow training modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from multimodal_datapipeline.utils.paths import package_root, workflow_root


BASELINE_NAMES = {
    "baseline_1_molecule_only",
    "baseline_2_protein_only",
    "baseline_3_image_only",
    "baseline_4_molecule_protein",
    "baseline_5_molecule_image",
    "baseline_6_protein_image",
    "baseline_7_molecule_protein_image",
}


def _load_train_module(baseline_name: str) -> ModuleType:
    if baseline_name not in BASELINE_NAMES:
        known = ", ".join(sorted(BASELINE_NAMES))
        raise ValueError(f"Unknown baseline {baseline_name!r}. Expected one of: {known}")

    baseline_dir = workflow_root() / "baselines" / baseline_name
    train_path = baseline_dir / "train.py"
    if not train_path.is_file():
        raise FileNotFoundError(f"Baseline training script not found: {train_path}")

    for path in (package_root(), baseline_dir):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    module_name = f"_mmdp_{baseline_name}_train"
    spec = importlib.util.spec_from_file_location(module_name, train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load baseline training script: {train_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_baseline(baseline_name: str) -> None:
    module = _load_train_module(baseline_name)
    main = getattr(module, "main", None)
    if main is None:
        raise AttributeError(f"{baseline_name}/train.py does not define main()")
    main()


def run_baseline_from_file(file_path: str | Path) -> None:
    baseline_name = Path(file_path).resolve().parent.name
    run_baseline(baseline_name)


def baseline_1_molecule_only() -> None:
    run_baseline("baseline_1_molecule_only")


def baseline_2_protein_only() -> None:
    run_baseline("baseline_2_protein_only")


def baseline_3_image_only() -> None:
    run_baseline("baseline_3_image_only")


def baseline_4_molecule_protein() -> None:
    run_baseline("baseline_4_molecule_protein")


def baseline_5_molecule_image() -> None:
    run_baseline("baseline_5_molecule_image")


def baseline_6_protein_image() -> None:
    run_baseline("baseline_6_protein_image")


def baseline_7_molecule_protein_image() -> None:
    run_baseline("baseline_7_molecule_protein_image")
