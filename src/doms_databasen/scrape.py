from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


url = "https://domsdatabasen.dk/#sag/45"

options = Options()
options.add_argument("start-maximized")
DATA_PATH = "/Users/oliver/doms_databasen/data/raw"
options.add_experimental_option(
    "prefs",
    {
        "download.default_directory": DATA_PATH,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,
    },
)


driver = webdriver.Chrome(
    service=ChromeService(ChromeDriverManager().install()), options=options
)

driver.get(url)


xpath = "//*[local-name() = 'svg'][@data-icon = 'download']"
element = driver.find_element(By.XPATH, xpath)


page_source = driver.page_source
print(page_source)
