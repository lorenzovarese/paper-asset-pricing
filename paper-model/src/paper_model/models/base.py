from __future__ import annotations
from abc import ABC, abstractmethod
import polars as pl
from typing import Any, Dict


class BaseModel(ABC):
    """Abstract base class for all asset pricing models."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.model = None  # To hold the trained model object (e.g., sklearn model, statsmodels results)
        self.evaluation_results: Dict[str, Any] = {}
        self.checkpoint_data: pl.DataFrame | None = None

    @abstractmethod
    def train(self, data: pl.DataFrame) -> None:
        """
        Trains the model using the provided data.

        Args:
            data: A Polars DataFrame containing the necessary features and target.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: pl.DataFrame) -> pl.Series:
        """
        Generates predictions using the trained model.

        Args:
            data: A Polars DataFrame containing the features for prediction.

        Returns:
            A Polars Series of predicted values.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, y_true: pl.Series, y_pred: pl.Series) -> Dict[str, Any]:
        """
        Evaluates the trained model and returns performance metrics.

        Args:
            y_true: A Polars Series of true target values.
            y_pred: A Polars Series of predicted target values.

        Returns:
            A dictionary of evaluation metrics.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"
