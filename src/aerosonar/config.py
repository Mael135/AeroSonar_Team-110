"""YAML configuration loading.

Configuration paths are resolved relative to the current working directory, so every
entry point in this project must be run from the repository root.
"""
import yaml
from pathlib import Path

DEFAULT_CONFIG_PATH = "src/aerosonar/config/default.yaml"
MODEL_CONFIG_PATH = "src/aerosonar/config/model.yaml"

def load_default_config():
    """Load the project's default configuration.

    Returns:
        dict: Parsed contents of ``default.yaml``.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    return load_config(DEFAULT_CONFIG_PATH)

def load_model_config():
    """Load the optional model-specific configuration overlay.

    Returns:
        dict: Parsed contents of ``model.yaml``.

    Raises:
        FileNotFoundError: If the overlay is not present.
    """
    return load_config(MODEL_CONFIG_PATH)

def load_config(config_path: str) -> dict:
    """Read and parse a single YAML configuration file.

    Args:
        config_path: Path to the file, relative to the working directory.

    Returns:
        dict: Parsed configuration.

    Raises:
        FileNotFoundError: If ``config_path`` does not name an existing file.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def merge_dict(a, b):
    """Recursively merge ``b`` into ``a``, modifying ``a`` in place.

    Nested dictionaries are merged key by key. Any other value in ``b`` replaces the
    corresponding entry in ``a``.

    Args:
        a: Destination dictionary, modified in place.
        b: Source dictionary, whose entries take precedence.

    Returns:
        dict: The merged dictionary ``a``.
    """
    for key, value in b.items():
        if isinstance(value, dict) and key in a and isinstance(a[key], dict):
            merge_dict(a[key], value)
        else:
            a[key] = value
    return a

def load_and_merge_configs(config_paths):
    """Load several YAML files and merge them in order.

    Args:
        config_paths: Iterable of paths. Later files override earlier ones.

    Returns:
        dict: The merged configuration.
    """
    cfg = {}
    for path in config_paths:
        cfg = merge_dict(cfg, load_config(path))
    return cfg
