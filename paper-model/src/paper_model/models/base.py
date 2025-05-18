from __future__ import annotations
from abc import ABC, abstractmethod
import polars as pl
from typing import Any, Dict, Tuple


class BaseModel(ABC):
    """Abstract base class for all asset pricing models."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.model = None  # To hold the trained model object (e.g., sklearn model)
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
    def evaluate(self, data: pl.DataFrame) -> Dict[str, Any]:
        """
        Evaluates the trained model and returns performance metrics.

        Args:
            data: A Polars DataFrame for evaluation (can be the same as training data or a test set).

        Returns:
            A dictionary of evaluation metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_checkpoint(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generates a checkpoint DataFrame containing model outputs
        (e.g., predicted returns, factor exposures) for downstream use.

        Args:
            data: The Polars DataFrame on which to generate predictions/outputs.

        Returns:
            A Polars DataFrame with checkpoint data.
        """
        raise NotImplementedError

    def run(self, data: pl.DataFrame) -> Tuple[Dict[str, Any], pl.DataFrame]:
        """
        Orchestrates the training, evaluation, and checkpoint generation for the model.

        Args:
            data: The Polars DataFrame to use for training and evaluation.

        Returns:
            A tuple containing:
            - A dictionary of evaluation results.
            - A Polars DataFrame of checkpoint data.
        """
        self.train(data)
        self.evaluation_results = self.evaluate(data)
        self.checkpoint_data = self.generate_checkpoint(data)
        return self.evaluation_results, self.checkpoint_data

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"
