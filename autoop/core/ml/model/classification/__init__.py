"""Import sub modules for use in parent module for get models."""

from autoop.core.ml.model.classification.linear_discriminant_analysis import (
    LinearDiscriminantAnalysis
)
from autoop.core.ml.model.classification.k_nearest_neighbors import (
    KNearestNeighbors
)
from autoop.core.ml.model.classification.logistic_regression import (
    LogisticRegression
)

print(f"{LinearDiscriminantAnalysis()}, {KNearestNeighbors()}, " +
      f"{LogisticRegression()} have been successfully imported")
