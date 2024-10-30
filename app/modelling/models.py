from autoop.core.ml.model import Model
from typing import Literal


def get_models(type: Literal["regresion", "classification"]) -> list[Model]:
    """
    Get list of models that fit the type of target feature.

    Arguments:
        Type (Literal["regresion", "classification"]): type of target colum.

    Returns:
        list of model that fit the target feature.
    """
    match type:
        case "classification":
            pass
        case "regresion":
            pass
