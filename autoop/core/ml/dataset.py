from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """A class to represent an ML dataset"""
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize Dataset class with given arbitrary amount of arguments and
        keyword arguments.

        Returns:
        None
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str,
                       asset_path: str, version: str = "1.0.0") -> "Dataset":
        """ Create a dataset from a pandas dataframe."""
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """ Read data from a given path """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save_artifact(self, data: pd.DataFrame) -> bytes:
        """ Save data to a given path """
        bytes = data.to_csv(index=False).encode()
        return super().save_artifact(bytes)
