import os
from pathlib import Path

ENV_VAR = "PAPER_TOOLS_DIR"


def get_data_root() -> Path:
    """
    Return the user‐specified data directory.

    Raises:
        KeyError: if the environment variable is not set.
    """
    try:
        root = Path(os.environ[ENV_VAR]).expanduser().resolve()
    except KeyError as e:
        raise RuntimeError(
            f"{ENV_VAR} must be set to the full path where data is stored."
        ) from e

    root.mkdir(parents=True, exist_ok=True)
    return root
