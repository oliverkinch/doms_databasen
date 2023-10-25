import json


def save_dict_to_json(dict_, file_path):
    with open(file_path, "w") as f:
        json.dump(dict_, f, indent=4)
