from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoop.core.ml.feature import Feature


def get_task_type(target_colum: "Feature") -> str:
    """
    Get the name of the task type according to target colum.

    Parameters
    ----------
    target_colum : Feature
        Target colum of for the ML pipeline

    Returns
    -------
    str:
        the task type name of the task based of target colum.
    """
    if target_colum is None:
        return "(No target selected.)"
    else:
        match target_colum.type:
            case "numerical":
                return "regression"
            case "categorical":
                return "classification"
