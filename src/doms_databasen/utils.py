import json
import jsonlines


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


def init_jsonl(file_name: str) -> None:
    """Initializes jsonl file.

    The function is used in the DataSetBuilder class to initialize
    the dataset file, if it does not already exist.

    Args:
        file_name (str):
            File name to initialize.
    """
    with open(file_name, "w") as _:
        pass


def append_jsonl(data: dict, file_name: str) -> None:
    """Appends data to jsonl file.

    Args:
        data (dict):
            Data to append.
        file_name (str):
            The name of the JSONL file where the data should be appended.
        keys_to_str (list):
            List of keys to convert to string (for example PosixPath objects)
    """

    with jsonlines.open(file_name, mode="a") as writer:
        writer.write(data)
