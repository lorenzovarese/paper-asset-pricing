"""paper-data package."""

__author__ = "Lorenzo Varese"
__version__ = "0.1.0"

from .manager import DataManager
from .run_pipeline import main

__all__ = ["DataManager", "main"]
