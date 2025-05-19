"""paper-portfolio package."""

__author__ = "Lorenzo Varese"
__version__ = "0.1.0"

from .manager import PortfolioManager
from .config_parser import load_config
from .run_pipeline import main

__all__ = ["PortfolioManager", "load_config", "main"]
