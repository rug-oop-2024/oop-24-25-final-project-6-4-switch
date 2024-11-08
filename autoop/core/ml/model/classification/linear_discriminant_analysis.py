import numpy as np
from autoop.core.ml.model.model import Model
from typing import Any


class LinearDiscriminantAnalysis(Model):
    """Class of linear discriminant analysis model."""

    def __init__(self) -> None:
        """
        Initialize model.

        Returns
        -------
        None
        """
        super().__init__()
        self._mean_vectors = {}
        self._priors = {}
        self._inv_cov_matrix = None

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
        classes = np.unique(labels)
        n_features = features.shape[1]
        cov_matrix = np.zeros((n_features, n_features))
        cov_matrix += [self._compute_class(c, features, labels)
                       for c in classes]
        cov_matrix /= (features.shape[0] - len(classes))
        self._inv_cov_matrix = np.linalg.inv(cov_matrix)

    def _compute_class(self, clas: Any, features: np.ndarray,
                       labels: np.ndarray) -> np.ndarray:
        """
        Compute the mean vectors and priors for a class.

        Parameters
        ----------
        clas : Any
            The class.
        features : ndarray
            Training features.
        labels : ndarray
            Training labels.

        Returns
        -------
        ndarray
            The computed matrix for the class.
        """
        class_features = features[labels == clas]
        self._mean_vectors[clas] = np.mean(class_features, axis=0)
        self._priors[clas] = class_features.shape[0] / features.shape[0]

        return np.cov(class_features, rowvar=False) * \
            (class_features.shape[0] - 1)

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
        scores = np.zeros((features.shape[0], len(self.mean_vectors)))

        for idx, c in enumerate(self._mean_vectors):
            mean_vector = self._mean_vectors[c]
            prior = self._priors[c]

            scores[:, idx] = (
                features @ self._inv_cov_matrix @ mean_vector.T -
                0.5 * mean_vector @ self._inv_cov_matrix @ mean_vector.T +
                np.log(prior)
            )

        return np.argmax(scores, axis=1)
