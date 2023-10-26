"""This scripts finalizes the data by merging all the processed data into a single dataset.

Usage:
    >>> python src/scripts/finalize.py

    Overwrite existing dataset:
    >>> python src/scripts/finalize.py 'finalize.force=True'
"""


import hydra
from pathlib import Path
from omegaconf import DictConfig
from logging import getLogger

from src.doms_databasen.utils import read_json, init_jsonl, append_jsonl


logger = getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    data_processed_dir = Path(cfg.paths.data_processed_dir)
    data_final_dir = Path(cfg.paths.data_final_dir)
    dataset_path = data_final_dir / cfg.file_names.dataset

    if dataset_path.exists() and not cfg.finalize.force:
        logger.info(
            f"Dataset already exists at {dataset_path}. Use 'finalize.force=True' to overwrite."
        )
        return

    logger.info("Initializing dataset with path: {dataset_path}")
    init_jsonl(dataset_path)

    processed_case_paths = [
        case_path for case_path in data_processed_dir.iterdir() if case_path.is_dir()
    ]
    logger.info(f"Found {len(processed_case_paths)} cases in {data_processed_dir}")

    for path in processed_case_paths:
        json_data = read_json(path / cfg.file_names.processed_data)
        append_jsonl(json_data, dataset_path)

    logger.info(f"Dataset saved at {dataset_path}")


if __name__ == "__main__":
    main()
