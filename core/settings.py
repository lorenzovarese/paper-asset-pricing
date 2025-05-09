from pathlib import Path
import os

# Base directory of project (two levels up from this file)
BASE_DIR = Path(__file__).parent.parent.resolve()

# Directory for user data (default to ./data)
DATA_DIR = Path(os.getenv("PAPER_DATA_PATH", BASE_DIR / "data"))

# Directory for cached intermediate files
CACHE_DIR = Path(os.getenv("PAPER_CACHE_PATH", BASE_DIR / "cache"))
