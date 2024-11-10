from autoop.core.ml.model.classification.naive_bayes import (
    NaiveBayes
)
from autoop.core.ml.model.classification.k_nearest_neighbors import (
    KNearestNeighbors
)
from autoop.core.ml.model.classification.logistic_regression import (
    LogisticRegression
)

print(f"{NaiveBayes()}, {KNearestNeighbors()}, " +
      f"{LogisticRegression()} have been successfully imported")
