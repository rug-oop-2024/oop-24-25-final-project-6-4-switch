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
        self.hyper_parameters["trees"] = trees
        self.hyper_parameters["depth"] = depth
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
        trees = self.hyper_parameters["trees"]
        depth = self.hyper_parameters["depth"]

        for _ in range(trees):
            indices = np.random.choice(features.shape[0], features.shape[0],
                                       replace=True)
            sample_features = features[indices]
            sample_labels = labels[indices]

            tree = DecisionTreeRegressor(max_depth=depth)

            # Train the tree on the sampled data
            tree.fit(sample_features, sample_labels)
            self.trees.append(tree)

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
