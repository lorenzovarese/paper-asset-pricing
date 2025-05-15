import abc
from pathlib import Path
import joblib  # type: ignore


class BaseModel(abc.ABC):
    """
    Abstract base class for all models.
    """

    @abc.abstractmethod
    def fit(self, X, y) -> "BaseModel": ...

    @abc.abstractmethod
    def predict(self, X): ...

    def save(self, path: Path) -> None:
        """
        Persist the fitted model to disk.
        """
        joblib.dump(self, path)
