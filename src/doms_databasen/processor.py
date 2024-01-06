import os
import re
from logging import getLogger
from pathlib import Path

from omegaconf import DictConfig

from .constants import N_FILES_PROCESSED_CASE_DIR, N_FILES_RAW_CASE_DIR
from .text_extraction import PDFTextReader
from .utils import read_json, save_dict_to_json

logger = getLogger(__name__)


class Processor(PDFTextReader):
    """Processor for scraped data from the DomsDatabasen website.

    Args:
        config (DictConfig):
            Config file

    Attributes:
        config (DictConfig):
            Config file
        data_raw_dir (Path):
            Path to raw data directory
        data_processed_dir (Path):
            Path to processed data directory
        force (bool):
            If True, existing data will be overwritten.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)
        self.config = config
        self.data_raw_dir = (
            Path(self.config.paths.data_raw_dir)
            if not self.config.testing
            else Path(self.config.paths.test_data_raw_dir)
        )
        self.data_processed_dir = Path(self.config.paths.data_processed_dir)
        self.force = self.config.process.force

    def process(self, case_id: str) -> None:
        """Processes a single case.

        This function takes the raw tabular data and
        adds the text from the pdf to it + the ID of the case.

        Args:
            case_id (str):
                Case ID
        """
        case_id = str(case_id)

        case_dir_raw = self.data_raw_dir / case_id
        case_dir_processed = self.data_processed_dir / case_id

        # Check if raw data for case ID exists.
        if not self._raw_data_exists(case_dir_raw):
            logger.info(f"{self.config.message_case_not_found} {case_id}")
            return

        # If case has already been processed, skip, unless force=True.
        if self._already_processed(case_dir_processed) and not self.force:
            logger.info(f"{self.config.message_case_already_processed} {case_id}")
            return

        # Process data for the case.
        logger.info(f"{self.config.message_processing_case} {case_id}")

        case_dir_processed.mkdir(parents=True, exist_ok=True)

        tabular_data = read_json(case_dir_raw / self.config.file_names.tabular_data)

        processed_data = {}
        processed_data["tabular_data"] = tabular_data
        processed_data["case_id"] = case_id
        pdf_path = case_dir_raw / self.config.file_names.pdf_document
        text_anonymized, pdf = self.extract_text(
            pdf_path=pdf_path,
        )

        processed_data["text_anon_tagged"] = text_anonymized
        processed_data["pdf"] = pdf

        # Remove anonymization tags from text.
        text = re.sub(r"<anonym.*</anonym>", "", text_anonymized)
        processed_data["text"] = text

        if not self.config.testing:
            save_dict_to_json(
                processed_data,
                case_dir_processed / self.config.file_names.processed_data,
            )

        logger.info(f"{self.config.message_case_done} {case_id}")

        # Return data for testing purposes.
        return processed_data

    def process_all(self) -> None:
        """Processes all cases in data/raw"""
        logger.info(self.config.message_processing_all_cases)
        case_ids = sorted(
            [
                case_path.name
                for case_path in self.data_raw_dir.iterdir()
                if case_path.is_dir()  # Exclude .gitkeep
            ],
            key=lambda case_id: int(case_id),
        )
        for case_id in case_ids:
            self.process(case_id)

    def _already_processed(self, case_dir) -> bool:
        """Checks if a case has already been processed.

        If a case has already been processed, the case directory will
        exist and will contain one file with the tabular data.

        Args:
            case_dir (Path):
                Path to case directory

        Returns:
            bool:
                True if case has already been processed. False otherwise.
        """
        return (
            case_dir.exists()
            and len(os.listdir(case_dir)) == N_FILES_PROCESSED_CASE_DIR
        )

    def _raw_data_exists(self, case_dir) -> bool:
        """Checks if raw data for a case exists.

        If a case has been scraped successfully, then the case directory exists
        and contains two files: the PDF document and the tabular data.

        Same code as the method `_already_scraped` from class `DomsDatabasenScraper`
        (src/doms_databasen/scraper.py).

        Args:
            case_dir (Path):
                Path to case directory

        Returns:
            bool:
                True if case has already been scraped. False otherwise.
        """
        return case_dir.exists() and len(os.listdir(case_dir)) == N_FILES_RAW_CASE_DIR
