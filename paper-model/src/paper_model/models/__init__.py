"""models package."""

__author__ = "Lorenzo Varese"
__version__ = "0.1.0"

from .base import BaseModel
from .fama_french import FamaFrench3FactorModel
from .linear_regression import SimpleLinearRegression

__all__ = ["BaseModel", "FamaFrench3FactorModel", "SimpleLinearRegression"]
