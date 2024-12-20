from autoop.core.ml.model import Model
from autoop.core.ml.model.classification import (
    KNearestNeighbors,
    NaiveBayes,
    LogisticRegression
)
from autoop.core.ml.model.regression import (
    DecisionTree,
    MultipleLinearRegression,
    RandomForest
)

from typing import Literal, Dict


def list_models(type: Literal["regression",
                              "classification"]) -> Dict[str, "Model"]:
    """
    Get list of models that fit the type of target feature.

    Arguments:
        Type (Literal["regression", "classification"]): type of target colum.

    Returns:
        list of model that fit the target feature.
    """
    match type:
        case "classification":
            return {"K Nearest Neighbors": KNearestNeighbors,
                    "Naive Bayes": NaiveBayes,
                    "Logistic Regression": LogisticRegression}
        case "regression":
            return {"Decision Tree": DecisionTree,
                    "Multiple Linear Regression": MultipleLinearRegression,
                    "Random Forest": RandomForest}
