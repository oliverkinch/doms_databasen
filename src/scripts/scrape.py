"""Script for scraping the DomsDatabasen website.

Examples usages:
    Scrape single case:
    >>> python src/scripts/scrape.py 'case_id=123'

    Scrape single case and overwrite existing data:
    >>> python src/scripts/scrape.py 'case_id=123' 'force=True'

    Scrape all cases:
    >>> python src/scripts/scrape.py

    Scrape all cases and overwrite existing data:
    >>> python src/scripts/scrape.py 'force=True'
"""

import logging
import hydra
from omegaconf import DictConfig

from src.doms_databasen.scraper import DomsDatabasenScraper
# Importing as a module, doesn't work when running as a script?
# from doms_databasen.scraper import DomsDatabasenScraper

logger = logging.getLogger(__name__)
@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    if (not cfg.case_id and not cfg.scrape_all) or (cfg.case_id and cfg.scrape_all):
        logger.info(cfg.messages.give_correct_inputs)
        exit()

    scraper = DomsDatabasenScraper(cfg=cfg)
    if cfg.scrape_all:
        logger.info("Scraping all cases")
        scraper.scrape_all(force=cfg.force)
    else:
        logger.info(f"Scraping case {cfg.case_id}")
        scraper.scrape_case(cfg.case_id, force=cfg.force)

    logger.info("Done!")


if __name__ == "__main__":
    main()
