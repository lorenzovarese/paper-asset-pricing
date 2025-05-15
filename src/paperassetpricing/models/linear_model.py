from sklearn.linear_model import LinearRegression  # type: ignore
from .base_model import BaseModel
from .model_registry import register_model


@register_model("ols")
class LinearModel(BaseModel):
    """
    Ordinary least squares via scikit-learn.
    """

    def __init__(self, **kwargs):
        # you could accept kwargs for fit parameters here
        self._model = LinearRegression(**kwargs)

    def fit(self, X, y) -> "LinearModel":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)
