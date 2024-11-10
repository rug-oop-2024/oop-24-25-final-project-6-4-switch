import numpy as np
from autoop.core.ml.model.model import Model


class MultipleLinearRegression(Model):
    """Class of multiple linear regression model."""

    def __init__(self, alpha: float = 0.0) -> None:
        """
        Initialize model.

        Parameters
        ----------
        alpha : float
            Regularization strength.

        Returns
        -------
        None
        """
        super().__init__()
        self.hyper_parameters = {"alpha": alpha}
        self.is_fitted = False
        self.type = "regression"
        self.name = "Multiple Linear Regression"

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit the model to the provided training data.

        Parameters
        ----------
        features : ndarray
            Training features.
        labels : ndarray
            Training labels.

        Returns
        -------
        None
        """
        squiggle = np.c_[features, np.ones(features.shape[0])]
        alpha = self.hyper_parameters["alpha"]

        try:
            inversed_matrix = np.linalg.inv(squiggle.T @ squiggle + alpha)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Matrix is not invertible with the" +
                                        "added one column at the end.")
        self.parameters = {"weights": inversed_matrix @ squiggle.T @ labels}
        self.is_fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Parameters
        ----------
        features : ndarray
            Test features.

        Returns
        -------
        ndarray
            Predicted values.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained, fit it first!")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        bias = np.c_[features, np.ones(features.shape[0])]

        _, column = bias.shape
        coef_row, _ = self.parameters["weights"].shape

        if column != coef_row:
            raise ValueError("Wrong size for the ndarray.")

        return bias @ self.parameters["weights"]
