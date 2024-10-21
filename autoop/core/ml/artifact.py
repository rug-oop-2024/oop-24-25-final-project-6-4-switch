from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    """Artifact.
    Atributes:
        name (str): name of artifect.
        asset_path (str): path of the asset.
        data (bytes): base64 encoded path to the data.
        version (str): version of the artifact.
        type (str): type of artifact.
        metadata (dict): dictionary with experiment and run id.
        tags (list): list of tags for the artifact.
    """
    name: str = Field()
    asset_path: str = Field(default=None)
    data: bytes = Field()
    version: str = Field(default="1.0.0")
    type: str = Field(default=None)
    metadata: dict = Field(default={"experiment_id": None,
                                    "run_id": None})
    tags: list[str] = Field(default=[])

    def read(self) -> bytes:
        return self.data
