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
        self.is_fitted: bool = False
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
        labels = np.argmax(labels, axis=1)
        n_samples, n_features = features.shape
        n_classes = len(np.unique(labels))

        weights = np.zeros((n_features, n_classes))
        bias = np.zeros(n_classes)

        for _ in range(self.hyper_parameters["iterations"]):
            linear_model = np.dot(features, weights) + bias
            predictions = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(features.T, (predictions))
            db = (1 / n_samples) * np.sum(predictions, axis=0)

            weights -= self.hyper_parameters["learning_rate"] * dw
            bias -= self.hyper_parameters["learning_rate"] * db

        self.parameters = {
            "weights": weights,
            "bias": bias
        }
        self.is_fitted = True

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

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained, fit it first!")

        linear_model = (np.dot(features, self.parameters["weights"]) +
                        self.parameters["bias"])
        probabilities = self._sigmoid(linear_model)

        return probabilities
