from typing import List, TYPE_CHECKING
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detect feature types for a ML dataset. Assume there are only categorical
    and numerical features, and that there are NaN values.

    Parameters
    ----------
    dataset : Dataset
        The dataset of the ML class.

    Returns
    -------
    List[Feature]
        List of features with their types.
    """
    data: pd.DataFrame = dataset.read()
    feature_list: list[Feature] = []

    for label in data.columns:
        data_type: str = None
        match data[label].dtype:
            # Match for numerical.
            case np.int64 | np.float64:
                data_type = "numerical"
            # Match for categorical.
            case _:
                data_type = "categorical"
        feature_list.append(Feature(name=label, type=data_type))

    return feature_list
