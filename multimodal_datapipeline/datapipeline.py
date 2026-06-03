"""CLI entrypoint for Phase 1 dataset ingestion and missing-data downloads."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multimodal_datapipeline.pipelines.dataset_pipeline import main


if __name__ == "__main__":
    main()
