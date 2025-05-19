"""models package."""

__author__ = "Lorenzo Varese"
__version__ = "0.1.0"

from .base import BaseModel
from .sklearn_model import SklearnModel

__all__ = ["BaseModel", "SklearnModel"]
