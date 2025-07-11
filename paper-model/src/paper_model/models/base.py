from __future__ import annotations
from abc import ABC, abstractmethod
import polars as pl
from typing import Any, TypeVar, Generic

ModelType = TypeVar("ModelType")


class BaseModel(ABC, Generic[ModelType]):
    """Abstract base class for all asset pricing models."""

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config
        self.model: ModelType | None = None
        self.evaluation_results: dict[str, Any] = {}
        self.checkpoint_data: pl.DataFrame | None = None

    @abstractmethod
    def train(
        self, train_data: pl.DataFrame, validation_data: pl.DataFrame | None = None
    ) -> None:
        """
        Trains the model using the provided data.

        Args:
            train_data: A Polars DataFrame for training.
            validation_data: An optional Polars DataFrame for hyperparameter tuning.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: pl.DataFrame) -> pl.Series:
        """Generates predictions using the trained model."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, y_true: pl.Series, y_pred: pl.Series) -> dict[str, Any]:
        """Evaluates the trained model and returns performance metrics."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"
