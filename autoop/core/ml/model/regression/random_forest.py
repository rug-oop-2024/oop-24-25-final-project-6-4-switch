import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.tree import DecisionTreeRegressor


class RandomForest(Model):
    """Class of random forest model."""

    def __init__(self, n_trees: int = 10, depth: int = None) -> None:
        """
        Initialize model.

        Parameters
        ----------
        trees : int
            Number of trees in the forest.
        depth : int
            Maximum depth per tree.

        Returns
        -------
        None
        """
        super().__init__()
        self.hyper_parameters = {
            "n_trees": n_trees,
            "depth": depth
        }
        self.parameters = {"trees": []}
        self.is_fitted = False
        self.type = "regression"
        self.name = "Random Forest"

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
            "trees": [self._fit_single(features, labels) for
                      _ in range(self.hyper_parameters["n_trees"])]}
        self.is_fitted = True

    def _fit_single(self, features: np.ndarray, labels: np.ndarray) \
            -> "DecisionTreeRegressor":
        """
        Fit the model individually.

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
        indices = np.random.choice(features.shape[0], features.shape[0],
                                   replace=True)
        sample_features = features[indices]
        sample_labels = labels[indices]

        tree = DecisionTreeRegressor(max_depth=self.hyper_parameters["depth"])

        return tree.fit(sample_features, sample_labels)

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

        tree_predictions = np.array([tree.predict(features) for tree in
                                     self.parameters["trees"]])
        predictions = np.mean(tree_predictions, axis=0)

        return predictions
