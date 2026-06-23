import yaml
from pathlib import Path

DEFAULT_CONFIG_PATH = "src/aerosonar/config/default.yaml"
MODEL_CONFIG_PATH = "src/aerosonar/config/model.yaml"

def load_default_config():
    return load_config(DEFAULT_CONFIG_PATH)

def load_model_config():
    return load_config(MODEL_CONFIG_PATH)

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def merge_dict(a, b):
    """Merge two dictionaries recursively."""
    for key, value in b.items():
        if isinstance(value, dict) and key in a and isinstance(a[key], dict):
            merge_dict(a[key], value)
        else:
            a[key] = value
    return a

def load_and_merge_configs(config_paths):
    cfg = {}
    for path in config_paths:
        cfg = merge_dict(cfg, load_config(path))
    return cfg