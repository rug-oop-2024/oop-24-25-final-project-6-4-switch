from typing import TYPE_CHECKING
from autoop.core.ml.dataset import Dataset

if TYPE_CHECKING:
    from autoop.core.ml.artifact import Artifact


def list_dataset(registry_list: list["Artifact"]) -> list["Dataset"]:
    """
    convert registry list to list of datasets.

    Arguments:
        registry_list (list[Artifacts]): registry list of base
        class artifact of dataset type.

    returns:
        List of datasets converted from registry list.
    """
    dataset_list: list["Dataset"] = []

    for artifact in registry_list:
        dataset_list.append(Dataset(name=artifact.name,
                                    asset_path=artifact.asset_path,
                                    data=artifact.data,
                                    version=artifact.version))

    return dataset_list
