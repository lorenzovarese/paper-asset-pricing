from sklearn.linear_model import LinearRegression, HuberRegressor  # type: ignore
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


@register_model("ols-huber")
class HuberLinearModel(BaseModel):
    """
    Robust linear regression using Huber loss via scikit-learn's HuberRegressor.

    This model optimizes a combination of squared and absolute loss,
    reducing sensitivity to outliers compared to ordinary least squares.
    """

    def __init__(
        self,
        epsilon: float = 1.35,
        max_iter: int = 100,
        alpha: float = 0.0001,
        warm_start: bool = False,
        fit_intercept: bool = True,
        tol: float = 1e-05,
        **kwargs,
    ):
        # Initialize the underlying HuberRegressor with given hyperparameters
        self._model = HuberRegressor(
            epsilon=epsilon,
            max_iter=max_iter,
            alpha=alpha,
            warm_start=warm_start,
            fit_intercept=fit_intercept,
            tol=tol,
            **kwargs,
        )

    def fit(self, X, y) -> "HuberLinearModel":
        """
        Fit the Huber regressor to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : HuberLinearModel
            Fitted estimator.
        """
        self._model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict using the Huber Regressor model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns predicted values.
        """
        return self._model.predict(X)
