from typing import List, Union, Any, Dict
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline():
    """Class for training and predicting pipeline."""

    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8,
                 ) -> None:
        """
        Initialize the pipeline class.

        Argument:
            metrics (List[Metrics]): list of metrics for in this pipeline.
            dataset (Dataset): Dataset for the this pipeline to fit.
            model (Model): Regression or prediction model for this pipeline.
            input_features (List[Feature]): List of features in dataset to
            use as independent variables.
            target_feature (Feature): Feature that this pipeline should target
            regressing or predicting
            split (float): The split of data to training and testing.

        Returns:
            None
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if (target_feature.type == "categorical" and
                model.type != "classification"):
            raise ValueError(
                "Model type must be classification for " +
                "categorical target feature")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature")

    def __str__(self) -> None:
        """
        String representation of the class.

        Argument:
            None

        Returns:
            None
        """
        return f"""
    Pipeline(
    model={self._model.type} {self._model.name},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        Return Model that is in the pipeline.

        Argument:
            None

        Returns:
            Model in the pipeline.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Used to get the artifacts generated during the pipeline
        execution to be saved
        """
        artifacts = []

        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name,
                                          data=data,
                                          version="encoder",
                                          asset_path=f"encoder{name}"))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name,
                                          data=data,
                                          version="scalar",
                                          asset_path=f"scalar{name}"))

        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                                  data=pickle.dumps(pipeline_data),
                                  version="data",
                                  asset_path="encoderpipeline_config"))
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """Registers artifact to an dictionary of artifacts.

        Arguments:
            name (str): Name od the artifact to be registered.
            artifact (Artifact): artifact that is to be registered.

        Returns:
            None
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features,
                                            self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector,
        # sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [data for (feature_name, data, artifact)
                               in input_results]

    def _split_data(self) -> None:
        # Split the data into training and testing sets
        split = self._split
        self._train_X = [vector[:int(split * len(vector))] for vector
                         in self._input_vectors]
        self._test_X = [vector[int(split * len(vector)):] for vector
                        in self._input_vectors]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self) -> Dict[str, Union[List, Any]]:
        """
        Execute the full machine learning workflow including preprocessing,
        data splitting, training, and evaluation.

        This method performs the following steps:
        1. Preprocessed features required for training.
        2. Splits the data into training and test sets.
        3. Trains the model on the training data.
        4. Evaluates the model on the test set.

        Returns
        -------
        dict[str, Union[List, Any]]
            A dictionary containing the training and evaluation metrics of the
            model, and the model's predictions on the test.
        """
        self._preprocess_features()
        self._split_data()
        self._train()

        train_x = self._compact_vectors(self._train_X)
        train_y = self._train_y
        train_predictions = self.model.predict(train_x)
        train_metric_results = []

        for metric in self._metrics:
            train_result = metric(train_predictions, train_y)
            train_metric_results.append((metric, train_result))

        self._evaluate()

        return {
            "train_metrics": train_metric_results,
            "metrics": self._metrics_results,
            "predictions": self._predictions
        }
