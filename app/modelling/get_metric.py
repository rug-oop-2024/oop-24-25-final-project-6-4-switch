from typing import Literal
from autoop.core.ml.metric import (
    Metric,
    MeanAbsoluteError,
    MeanSquaredError,
    RSquared,
    Accuracy,
    Precision,
    LogLoss)


def get_metrics(type: Literal["regression", "classification"]) -> list[Metric]:
    """
    Get list of models that fit the type of target feature.

    Arguments:
        Type (Literal["regression", "classification"]): type of target colum.

    Returns:
        list of model that fit the target feature.
    """
    match type:
        case "classification":
            return [MeanAbsoluteError(),
                    MeanSquaredError(),
                    RSquared()]
        case "regression":
            return [Accuracy(),
                    Precision(),
                    LogLoss()]
