import json
from hydra import compose, initialize
from omegaconf import DictConfig


def save_dict_to_json(dict_, file_path):
    with open(file_path, "w") as f:
        json.dump(dict_, f, indent=4)


def read_config(config_path, config_name) -> DictConfig:
    """Reads the config file.

    Returns:
        DictConfig:
            Config file
    """
    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)
    return cfg
