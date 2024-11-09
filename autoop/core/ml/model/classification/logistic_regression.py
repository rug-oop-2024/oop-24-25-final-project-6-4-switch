import numpy as np
from autoop.core.ml.model.model import Model


class LogisticRegression(Model):
    """Class of logistic regression model."""

    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000) \
            -> None:
        """
        Initialize model.

        Parameters
        ----------
        learning_rate : float
            Learning rate for gradient descent.
        iterations : int
            Number of iterations for gradient descent.

        Returns
        -------
        None
        """
        super().__init__()
        self.hyper_parameters = {
            "learning_rate": learning_rate,
            "iterations": iterations
        }
        self._weights = None
        self._bias = 0
        self.type = "classification"
        self.name = "Logistic Regression"

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
        n_samples, n_features = features.shape
        self._weights = np.zeros(n_features)
        self._bias = 0

        learning_rate = self.hyper_parameters["learning_rate"]
        iterations = self.hyper_parameters["iterations"]

        for _ in range(iterations):
            linear_model = np.dot(features, self._weights) + self._bias
            predictions = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(features.T, (predictions - labels))
            db = (1 / n_samples) * np.sum(predictions - labels)
            self._weights -= learning_rate * dw
            self._bias -= learning_rate * db

    def _sigmoid(self, linear_model: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid function

        Parameters
        ----------
        linear_model : ndarray
            Linear model to computer the sigmoid function.

        Returns
        -------
        ndarray
            Predictions based on the sigmoid function computation.
        """
        return 1 / (1 + np.exp(-linear_model))

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
        """
        linear_model = np.dot(features, self._weights) + self._bias
        probabilities = self._sigmoid(linear_model)
        return (probabilities >= 0.5).astype(int)
