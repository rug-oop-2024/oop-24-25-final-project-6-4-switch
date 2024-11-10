from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    """
    Feature class representing a feature in a ML model.

    Attributes
    ----------
    name : str
        Name of feature.
    type : str
        Type of feature.
    """
    name: str = Field(..., description="The name of the feature.")
    type: Literal["numerical", "categorical"] = \
        Field(..., description="The type of feature")

    def __str__(self) -> None:
        """
        Return string representation of the feature class.

        Returns
        -------
        None
        """
        return f"Feature \"{self.name}\" is of type {self.type}."
