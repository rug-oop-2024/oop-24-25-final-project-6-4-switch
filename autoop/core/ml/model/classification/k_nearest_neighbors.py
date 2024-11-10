from collections import Counter
import numpy as np
from autoop.core.ml.model.model import Model


class KNearestNeighbors(Model):
    """Class of KNN (K Nearest Neighbors) model."""

    def __init__(self, k: int = 3) -> None:
        """
        Initialize model.

        Parameters
        ----------
        k : int
            k value - the amount of neighbors chosen.

        Returns
        -------
        None
        """
        super().__init__()
        self.hyper_parameters = {"k": k}
        self.is_fitted = False
        self.type = "classification"
        self.name = "K Nearest Neighbors"

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
        self.parameters = {
            "features": features,
            "labels": np.argmax(labels, axis=1)
        }
        self.is_fitted = True

    def predict(self, features: np.ndarray) -> np.array:
        """
        Make predictions using the fitted model.

        Parameters
        ----------
        features : ndarray
            Test features.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained, fit it first!")

        return np.array([self._predict_single(feature) for feature
                         in features])

    def _predict_single(self, feature: np.ndarray) -> np.str_:
        """
        Predict the truth from a single array.

        Parameters
        ----------
        feature : ndarray
            Points in an n dimension.

        Returns
        -------
        str_
            Predicted label.
        """
        distance = np.linalg.norm(self.parameters["features"] -
                                  feature, axis=1)

        indices = np.argsort(distance)[:self.hyper_parameters["k"]]

        nearest_labels = [self.parameters["labels"][i] for i in indices]

        label_counts = Counter(nearest_labels)

        unique_labels = np.unique(self.parameters["labels"])

        probabilities = np.zeros(len(unique_labels))

        total_neighbors = self.hyper_parameters["k"]
        for label, count in label_counts.items():
            label_index = np.where(unique_labels == label)[0][0]
            probabilities[label_index] = count / total_neighbors

        return probabilities
