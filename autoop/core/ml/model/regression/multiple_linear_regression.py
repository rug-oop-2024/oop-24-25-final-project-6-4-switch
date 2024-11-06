import numpy as np
from autoop.core.ml.model.model import Model


class MultipleLinearRegression(Model):
    """Class of multiple linear regression model."""

    def __init__():
        """
        Initialize model.

        Returns
        -------
        None
        """
        super().__init__()

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
        squiggle = np.c_[features, np.ones(features.shape[0])]

        try:
            inversed_matrix = np.linalg.inv(squiggle.T @ squiggle)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Matrix is not invertible with the" +
                                        "added one column at the end.")
        self._parameters["weights"] = inversed_matrix @ squiggle.T @ labels

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
        if "weights" not in self.parameters:
            raise ValueError("Model has not been trained." +
                             "Please call the fit method first.")

        if features.ndim == 1:
            observations = features.reshape(1, -1)

        observations_bias = np.c_[observations, np.ones(observations.shape[0])]

        _, column = observations_bias.shape
        coef_row, _ = self.parameters["coef"].shape

        if column != coef_row:
            raise ValueError("Wrong size for the ndarray.")

        return observations_bias @ self.parameters["weights"]
