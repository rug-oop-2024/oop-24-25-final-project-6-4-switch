import numpy as np

from abc import abstractmethod, ABC
from copy import deepcopy
from pydantic import PrivateAttr
from typing import Dict

from autoop.core.ml.artifact import Artifact


class Model(ABC, Artifact):
    """Base class for regression and classification models."""

    _parameters: dict = PrivateAttr(default={})
    _hyper_parameters: dict = PrivateAttr(default={})
    _is_fitted: bool = PrivateAttr(default=False)

    def __init__(self) -> None:
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
    def parameters(self) -> Dict[str, np.ndarray]:
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

        Raises
        ------
        TypeError
            If hyper_parameters is not a dictionary
        """
        if not isinstance(parameters, dict):
            raise TypeError("Parameters must be a dictionary.")
        self._parameters = parameters

    @property
    def hyper_parameters(self) -> Dict[str, float]:
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

        Raises
        ------
        TypeError
            If hyper_parameters is not a dictionary
        """
        if not isinstance(hyper_parameters, dict):
            raise TypeError("Hyperparameters must be a dictionary.")
        self._hyper_parameters = hyper_parameters

    @property
    def is_fitted(self) -> bool:
        """
        Get bool if model is fitted.

        Returns
        -------
        bool
            False if model is not fitted, True if it is.
        """
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self, is_fitted: bool) -> None:
        """
        Set if model is fitted.

        Parameters
        ----------
        is_fitted : bool
            Boolean to set.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If is_fitted is not a bool.
        """
        if not isinstance(is_fitted, bool):
            raise TypeError("is_fitted must be a bool.")
        self._is_fitted = is_fitted

    def to_artifact(self, name: str) -> "Model":
        """
        Prepare the model to be an artifact.

        Returns:
        the model itself with the artifact attributes filled out.
        """
        self.name = name
        self.data = {"parameters": self._parameters,
                     "hyperparameters": self._hyper_parameters}

        return self
