from abc import ABC, abstractmethod
# TODO: Unsure what to use typing for
from typing import Any
import numpy as np

METRICS = [
    # Regression
    "mean_squared_error",
    "mean_absolute_error",
    "r_squared",
    # Classification
    "accuracy",
    "precision",
    "log_loss"
]


def get_metric(name: str) -> "Metric":
    """
    Return metric instance given by name.

    Parameters
    ----------
    name : str
        Metric name.

    Returns
    -------
    any
        Metric instance matching name.

    Raises
    ------
    ValueError
        If the name matches no instance or is incorrectly spelt.
    NotImplementedError
        If the name matches the instance but is not implemented.
    """
    if name in METRICS:
        # Parse string to match the class name.
        class_name: str = \
            ''.join(word.capitalize() for word in name.split('_'))
        try:
            # Return class
            return globals()[class_name]()
        except TypeError:
            raise NotImplementedError(f"{class_name} is not implemented.")
    else:
        raise ValueError(f"{class_name} is possibly mispelt.")


class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def __call__(self, truth: np.ndarray, pred: np.ndarray) -> float:
        """
        Calculate metric.

        Parameters
        ----------
        truth : ndarray
            Grouth truth values.
        pred : ndarray
            Predicted values.

        Returns
        -------
        float
            Real number representing the metric value.
        """
        ...


# region Regression
class MeanSquaredError(Metric):
    """Mean Squared Error metric implementation for regression."""

    def __call__(self, truth: np.ndarray, pred: np.ndarray):
        return np.mean((truth - pred) ** 2)


class MeanAbsoluteError(Metric):
    """Mean Absolute Error (MAE) metric implementation for regression."""

    def __call__(self, truth: np.ndarray, pred: np.ndarray) -> float:
        return np.mean(np.abs(truth - pred))


class RSquared(Metric):
    """R-Squared (R^2) metric implementation for regression."""

    def __call__(self, truth: np.ndarray, pred: np.ndarray) -> float:
        sum_squares = np.sum((truth - pred) ** 2)
        sum_total = np.sum((truth - np.mean(truth)) ** 2)

        return 1 - (sum_squares / sum_total)
# endregion


# region Classification
class Accuracy(Metric):
    """Accuracy metric implementation for classification."""

    def __call__(self, truth: np.ndarray, pred: np.ndarray):
        return np.mean(truth == pred)


class Precision(Metric):
    """Precision metric implementation for classification."""

    def __call__(self, truth: np.ndarray, pred: np.ndarray) -> float:
        true_pos = np.sum((truth == 1) & (pred == 1))
        false_pos = np.sum((truth == 0) & (pred == 1))

        return true_pos / (true_pos + false_pos)


class LogLoss(Metric):
    """Logarithmic Loss implemenation for classication."""

    def __call__(self, truth: np.ndarray, pred: np.ndarray) -> float:
        # Prevent taking log of 0 by clipping the array
        pred_clipped = np.clip(pred, 1e-15, 1 - 1e-15)

        return -np.mean(truth * np.log(pred_clipped) + (1 - truth) *
                        np.log(1 - pred_clipped))
# endregion
