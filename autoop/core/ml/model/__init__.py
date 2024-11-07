from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import (
    DecisionTree,
    MultipleLinearRegression,
    RandomForest
)
from autoop.core.ml.model.classification import (
    LinearDiscriminantAnalysis,
    KNearestNeighbors,
    LogisticRegression
)

REGRESSION_MODELS = [
    "decision_tree",
    "multiple_linear_regression",
    "random_forest"
]

CLASSIFICATION_MODELS = [
    "gradient_boosting",
    "k_nearest_neighbors",
    "logistic_regression"
]


def get_model(model_name: str) -> "Model":
    """
    Return model instance given by name.

    Parameters
    ----------
    model_name : str
        Model name.

    Returns
    -------
    Model
        Model instance matching the name.

    Raises
    ------
    ValueError
        If the name matches no instance or is incorrectly spelt.
    NotImplementedError
        If the name matches the instance but is not implemented.
    """
    if model_name in REGRESSION_MODELS or model_name in CLASSIFICATION_MODELS:
        try:
            match model_name:
                case "decision_tree":
                    return DecisionTree()
                case "multiple_linear_regression":
                    return MultipleLinearRegression()
                case "random_forest":
                    return RandomForest()
                case "linear_discriminant_analysis":
                    return LinearDiscriminantAnalysis()
                case "k_nearest_neighbors":
                    return KNearestNeighbors()
                case "logistic_regression":
                    return LogisticRegression()
        except TypeError:
            class_name: str = \
                ''.join(word.capitalize() for word in model_name.split('_'))
            raise NotImplementedError(f"{class_name} is not implemented.")
    else:
        raise ValueError(f"{model_name} is possibly mispelt.")
