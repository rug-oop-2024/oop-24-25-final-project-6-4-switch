from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import (
    DecisionTree,
    MultipleLinearRegression,
    RandomForest
)
from autoop.core.ml.model.classification import (
    NaiveBayes,
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


def get_model(name: str) -> "Model":
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
    if name in REGRESSION_MODELS or name in CLASSIFICATION_MODELS:
        model_name: str = \
            ''.join(word.capitalize() for word in name.split('_'))
        try:
            # Return class
            return globals()[model_name]()
        except TypeError:
            raise NotImplementedError(f"{name} is not implemented.")
    else:
        raise ValueError(f"{name} is possibly mispelt.")
