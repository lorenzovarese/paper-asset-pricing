"""paper-model package."""

__author__ = "Lorenzo Varese"
__version__ = "0.1.0"

from .manager import ModelManager
from .run_pipeline import main

__all__ = ["ModelManager", "main"]
