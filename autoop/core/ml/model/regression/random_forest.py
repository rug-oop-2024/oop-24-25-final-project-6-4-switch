import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.tree import DecisionTreeRegressor


class RandomForest(Model):
    """Class of random forest model."""

    def __init__(self, trees: int = 10, depth: int = None) -> None:
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
        self._decision_tree = DecisionTreeRegressor()
        self.hyper_parameters = {
            "trees": trees,
            "depth": depth
        }
        # TODO doesn't see trees as a field and refuses to assign []
        self.trees = []

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
        length = self.hyper_parameters["trees"]
        self.trees.append(self._fit_single(features, labels)
                          for _ in range(length))

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
        depth = self.hyper_parameters["depth"]
        indices = np.random.choice(features.shape[0], features.shape[0],
                                   replace=True)
        sample_features = features[indices]
        sample_labels = labels[indices]

        tree = DecisionTreeRegressor(max_depth=depth)

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
        """
        if not self.trees:
            raise ValueError("Model is not fitted yet." +
                             "Please call the fit method first.")

        tree_predictions = np.array([tree.predict(features) for tree in
                                     self.trees])
        predictions = np.mean(tree_predictions, axis=0)

        return predictions
