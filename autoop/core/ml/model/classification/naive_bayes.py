import numpy as np
from autoop.core.ml.model.model import Model


class NaiveBayes(Model):
    """Class of naive bayes model."""

    def __init__(self, smoothing: float = 1.0) -> None:
        """
        Initialize model.

        Parameters
        ----------
        smoothing : float
            Laplace smoothing parameter.

        Returns
        -------
        None
        """
        super().__init__()
        self.hyper_parameters = {"smoothing": smoothing}
        self.is_fitted = False
        self.type = "classification"
        self.name = "Naive Bayes"

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
        classes, class_counts  = np.unique(labels, return_counts=True)
        feature_probs = []

        for c in classes:
            class_data = features[labels == c]
            feature_prob = \
                (class_data.sum(axis=0) + self.hyper_parameters["smoothing"]) \
                / (class_data.shape[0] + self.hyper_parameters["smoothing"])
            feature_probs.append(feature_prob)

        self.parameters = {
            "class_probabilities": class_counts / len(labels),
            "feature_probabilities": np.array(feature_probs)
        }
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

        log_probs = []
        for i, class_prob in enumerate(self._parameters
                                       ["class_probabilities"]):
            log_prob = np.log(class_prob)
            log_prob += np.sum(
                np.log(self.parameters["feature_probabilities"][i]) *
                features, axis=1)
            log_probs.append(log_prob)

        return np.array(log_probs).T
