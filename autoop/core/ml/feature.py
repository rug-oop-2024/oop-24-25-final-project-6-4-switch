
from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    """Feature.
        name (str): name of feature.
        type (str): type of feature.
    """
    # attributes here
    name: str = Field()
    type: Literal["numerical", "categorical"] = Field()

    def __str__(self) -> None:
        
        return f"Column {self.name} is {self.type}"
