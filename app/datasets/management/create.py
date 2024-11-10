from autoop.core.ml.dataset import Dataset
import pandas as pd
from typing import IO


def create(file: IO, version: str) -> Dataset:
    """
    Get uploaded file and converts it to a dataset.

    Arguments:
        file (IO): uploaded file from st file uploader
        version (str): version of the dataset

    returns:
        dataset (Dataset)
    """
    if version == "":
        dataset = Dataset.from_dataframe(
            pd.read_csv(file),
            file.name[:-4] + "_1.0.0",
            asset_path=f"{file.name[:-4]}_1.0.0.csv")
    else:
        dataset = Dataset.from_dataframe(
            pd.read_csv(file),
            file.name[:-4] + f"_{version}",
            asset_path=f"{file.name[:-4]}_{version}.csv",
            version=version)

    return dataset
