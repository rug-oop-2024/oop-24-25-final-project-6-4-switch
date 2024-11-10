"""Import sub modules for use in parent module for get models."""

from autoop.core.ml.model.regression.decision_tree import (
    DecisionTree
)
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression
)
from autoop.core.ml.model.regression.random_forest import (
    RandomForest
)

DecisionTree(), MultipleLinearRegression(), RandomForest()
