import json
from tika import parser


def save_dict_to_json(dict_, file_path) -> None:
    """Saves a dictionary to a json file.

    Args:
        dict_ (dict):
            Dictionary to save
        file_path (Path):
            Path to json file
    """
    with open(file_path, "w") as f:
        json.dump(dict_, f, indent=4)


def read_json(file_path) -> None:
    """Reads a json file.

    Args:
        file_path (Path):
            Path to json file
    """
    with open(file_path, "r") as f:
        return json.load(f)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from pdf using the tika module.

    Args:
        pdf_path (Path):
            Path to pdf

    Returns:
        str:
            Text from pdf
    """
    text = parser.from_file(str(pdf_path))["content"]
    return text.strip()
