"""Script for Processing scraped data from the DomsDatabasen website.

Examples usages:
    Process single case:
    >>> python src/scripts/process.py 'process.case_id=123'

    Process single case and overwrite existing data:
    >>> python src/scripts/process.py 'case_id=123' 'force=True'

    Process all cases:
    >>> python src/scripts/process.py 'process.force=True'

    Process all cases and overwrite existing data:
    >>> python src/scripts/process.py 'process.force=True' 'process.all=True'
"""

import logging
import hydra
from omegaconf import DictConfig

from src.doms_databasen.processor import Processor

# Importing as a module, doesn't work when running as a script?
# from doms_databasen.processor import DomsDatabasenScraper

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    processor = Processor(cfg=cfg)
    if cfg.process.all:
        processor.process_all()
    elif cfg.process.case_id:
        processor.process(cfg.process.case_id)
    else:
        logger.info(cfg.process.messages.give_correct_inputs)

    logger.info("Done!")


if __name__ == "__main__":
    main()
