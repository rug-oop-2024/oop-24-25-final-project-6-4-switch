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
            "labels": labels
        }
        self.is_fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
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

        return np.ndarray([self._predict_single(feature) for feature
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
        distance = np.linalg.norm(self.parameters["features"] - feature,
                                  axis=1)
        indices = np.argsort(distance)[: self.hyper_parameters["k"]]
        nearest_label = [self.parameters["labels"][i][0]
                         for i in indices]
        most_common = Counter(nearest_label).most_common()

        return most_common[0][0]
