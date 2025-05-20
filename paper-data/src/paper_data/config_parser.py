from pathlib import Path
import yaml


def load_config(config_path: str | Path) -> dict:
    """
    Loads and parses a YAML configuration file.

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        A dictionary representing the parsed YAML configuration.

    Raises:
        FileNotFoundError: If the config_path does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
        ValueError: If the configuration file is empty or invalid.
    """
    config_path = Path(config_path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(
                f"Error parsing YAML file {config_path}: {exc}"
            ) from exc

    # Check if the loaded config is a dictionary. This handles empty files.
    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file '{config_path}' is empty or does not contain a valid YAML mapping (dictionary)."
        )

    return config
