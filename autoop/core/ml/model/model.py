from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal, Any
from pydantic import Field


class Model(ABC, Artifact):
    """Base class for regression and clasification models."""

    _parameters: dict = Field(default={np.ndarray})
    _hyper_parameters: dict = Field(default={float})

    def __init__():
        """
        Initialize model.

        Returns
        -------
        None
        """
        super().__init__()

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @property
    def parameters(self) -> dict[str: np.ndarray]:
        """
        Get strict parameters essential for prediction.

        Returns
        -------
        dict[str: ndarray]
            Parameters.
        """
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, parameters: dict[str: np.ndarray]) -> None:
        """
        Set strict parameters.

        Parameters
        ----------
        parameters : dict[str: ndarray]
            Parameters to set.

        Returns
        -------
        None
        """
        self._parameters = parameters

    @property
    def hyper_parameters(self) -> dict[str: float]:
        """
        Get hyperparameters useful for training.

        Returns
        -------
        dict[str: float]
            Hyperparameters.
        """
        return deepcopy(self._hyper_parameters)

    @hyper_parameters.setter
    def hyper_parameters(self, hyper_parameters: dict[str: float]) -> None:
        """
        Set hyperparameters.

        Parameters
        ----------
        hyper_parameters : dict[str: float]
            Hyperparameters.

        Returns
        -------
        None
        """
        self._hyper_parameters = hyper_parameters

    # TODO: Encode to base64 bytes for data
    def save(self, file_name: str) -> None:
        """
        Save your model.

        Parameters
        ----------
        file_name : str
            Name of the file.

        Returns
        -------
        None
        """

        

        self.data = {
            "parameters": self.parameters,
            "hyperparameters": self.hyper_parameters
        }
