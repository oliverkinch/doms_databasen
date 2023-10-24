import logging
import os
import shutil
import time
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from .constants import XPATHS, XPATHS_TABULAR_DATA
from .exceptions import PDFDownloadException
from .utils import save_dict_to_json

logger = logging.getLogger(__name__)


class DomsDatabasenScraper:
    """Scraper for domsdatabasen.dk"""

    def __init__(self, cfg) -> None:
        """Initializes the scraper.

        Args:
            cfg (DictConfig):
                Config file

        Attributes:
            cfg (DictConfig):
                Config file
            test_dir (Path):
                Path to test directory
            download_dir (Path):
                Path to download directory
            data_raw_dir (Path):
                Path to raw data directory
            driver (webdriver.Chrome):
                Chrome webdriver
        """
        self.cfg = cfg
        self.test_dir = Path(self.cfg.paths.test_dir)
        self.download_dir = (
            Path(self.cfg.paths.download_dir) if not self.cfg.testing else self.test_dir
        )
        self.data_raw_dir = Path(self.cfg.paths.data_raw_dir)

        self.intialize_downloader_folder()
        self.driver = self.start_driver()

    def scrape_case(self, case_id: str, force=False) -> bool:
        """Scrapes a single case from domsdatabasen.dk

        Args:
            case_id (str):
                Case ID
            force (bool, optional):
                If True, overwrites existing data. Defaults to False.

        Returns:
            bool:
                True if case has successfully been scraped or already
                has been scraped. False if case does not exist.
        """
        case_url = f"{self.cfg.domsdatabasen.url}/{case_id}"
        self.driver.get(case_url)
        # Wait for page to load
        time.sleep(1)
        self._accept_cookies()
        time.sleep(1)
        if not self._case_id_exists():
            logger.info(f"Case {case_id} does not exist")
            return False

        case_dir = (
            self.data_raw_dir / case_id
            if not self.cfg.testing
            else self.test_dir / self.cfg.test_case_name
        )
        if case_dir.exists():
            if force:
                shutil.rmtree(case_dir)
            else:
                logger.info(f"Case {case_id} already scraped. Use --force to overwrite")
                return True

        case_dir.mkdir(parents=True)

        self._download_pdf(case_dir)
        tabular_data = self._get_tabular_data()
        save_dict_to_json(tabular_data, case_dir / self.cfg.file_names.tabular_data)

        return True

    def scrape_all(self, force=False) -> None:
        """Scrapes all cases from domsdatabasen.dk

        Args:
            force (bool, optional):
                If True, overwrites existing data. Defaults to False.
        """
        case_id = 1
        while True:
            success = self.scrape_case(str(case_id), force=force)
            if not success:
                break
            case_id += 1

    def start_driver(self) -> webdriver.Chrome:
        """Starts a Chrome webdriver.

        Returns:
            webdriver.Chrome:
                Chrome webdriver
        """
        options = Options()

        options.add_experimental_option(
            "prefs",
            {
                "download.default_directory": os.path.abspath(self.download_dir),
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "plugins.always_open_pdf_externally": True,
            },
        )
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--headless")

        driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()), options=options
        )
        return driver

    def intialize_downloader_folder(self) -> None:
        """Initializes the download folder.

        Deletes the download folder if it exists and creates a new one.
        """
        if self.download_dir.exists():
            shutil.rmtree(self.download_dir)
        self.download_dir.mkdir()

    def _wait_download(self, files_before: set, timeout: int = 10) -> str:
        """Waits for a file to be downloaded to the download directory.

        Args:
            files_before (set):
                Set of file names in download folder before download.
            timeout (int, optional):
                Number of seconds to wait before timing out. Defaults to 10.

        Returns:
            file_name (str):
                Name of downloaded file (empty string if timeout)
        """
        time.sleep(1)
        endtime = time.time() + timeout
        while True:
            files_now = set(os.listdir(self.download_dir))
            new_files = files_now - files_before
            if len(new_files) == 1:
                file_name = new_files.pop()
                return file_name
            if time.time() > endtime:
                file_name = ""
                return file_name

    def _download_pdf(self, case_dir: Path) -> None:
        """Downloads the PDF document of the case.

        Args:
            case_dir (Path):
                Path to case directory
        """
        files_before_download = set(os.listdir(self.download_dir))

        download_element = WebDriverWait(self.driver, self.cfg.sleep).until(
            EC.presence_of_element_located((By.XPATH, XPATHS["download_pdf"]))
        )

        download_element.click()
        file_name = self._wait_download(files_before_download)
        if file_name:
            from_ = (
                self.download_dir / file_name
                if not self.cfg.testing
                else self.test_dir / file_name
            )
            to_ = case_dir / self.cfg.file_names.pdf_document
            shutil.move(from_, to_)
        else:
            raise PDFDownloadException()

    def _get_tabular_data(self) -> dict:
        """Gets the tabular data from the case.

        Returns:
            tabular_data (dict):
                Tabular data
        """
        self.driver.find_element(By.XPATH, XPATHS["Øvrige sagsoplysninger"]).click()
        # Wait for section to expand
        time.sleep(1)
        tabular_data = {}
        for key, xpath in XPATHS_TABULAR_DATA.items():
            element = self.driver.find_element(By.XPATH, xpath)
            tabular_data[key] = element.text.strip()

        return tabular_data

    def _case_id_exists(self) -> bool:
        """Checks if the case exists.

        If a case does not exist, the page will contain the text "Fejlkode 404". This
        is used to check if the case exists.

        Returns:
            bool:
                True if case exists. False otherwise.
        """
        try:
            _ = self.driver.find_element(By.XPATH, XPATHS["Fejlkode 404"])
            return False
        except NoSuchElementException:
            return True
        except Exception as e:
            logger.error(e)
            raise e

    def _accept_cookies(self) -> None:
        """Accepts cookies on the page."""
        element = WebDriverWait(self.driver, self.cfg.sleep).until(
            EC.presence_of_element_located((By.XPATH, XPATHS["Accept cookies"]))
        )
        element.click()
