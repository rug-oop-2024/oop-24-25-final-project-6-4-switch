
from typing import List, TYPE_CHECKING
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    data: pd.DataFrame = dataset.read()
    feature_list: list[Feature] = []
    for label in data.columns:
        data_type: str = None
        match data[label].dtype:
            case np.int64 | np.float64:
                data_type = "numerical"
            case _:
                data_type = "categorical"
        feature_list.append(Feature(name=label, type=data_type))

    return feature_list
