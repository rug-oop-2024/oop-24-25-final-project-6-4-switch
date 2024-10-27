from autoop.core.ml.model.classification import *
from autoop.core.ml.model.regression import *
from autoop.core.ml.model import Model


def get_models(type: str) -> list[Model]:
    match type:
        case "classification":
            pass
        case "regresion":
            pass
