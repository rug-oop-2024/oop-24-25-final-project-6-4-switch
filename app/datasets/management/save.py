from autoop.core.ml.dataset import Dataset
from app.core.system import AutoMLSystem


def save(dataset: Dataset) -> bool:
    """
    Save the uploaded dataset.

    arguments:
        dataset (Dataset): dataset that needs to be saved.

    returns:
        bool: if the saving was succesfull or not.
    """
    automl = AutoMLSystem.get_instance()
    automl.registry.register(dataset)
    return True
