import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.tree import DecisionTreeRegressor


class DecisionTree(Model):
    """
    Class of decision tree model, using the DecisionTreeRegressor module from
    scikit.
    """

    def __init__(self, depth: int = None) -> None:
        """
        Initialize model.

        Parameters
        ----------
        depth : int
            Maximum depth of the tree.

        Returns
        -------
        None
        """
        super().__init__()
        self.hyper_parameters["depth"] = depth
        self._tree = DecisionTreeRegressor(max_depth=depth)

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
        self.tree.fit(features, labels)

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
        return self.tree.predict(features)
