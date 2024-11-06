from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    """
    Artifact.

    Atributes:
        name (str): name of artifect.
        asset_path (str): path of the asset.
        data (bytes): base64 encoded path to the data.
        version (str): version of the artifact.
        type (str): type of artifact.
        metadata (dict): dictionary with experiment and run id.
        tags (list): list of tags for the artifact.
    """
    name: str = Field(default=None)
    asset_path: str = Field(default=None)
    data: bytes = Field(default=None)
    version: str = Field(default="1.0.0")
    type: str = Field(default=None)
    metadata: dict = Field(default={"experiment_id": None,
                                    "run_id": None})
    tags: list[str] = Field(default=[])

    def read(self) -> bytes:
        """
        Return data saved in this artifact.

        Arguments:
            None

        Returns:
            None
        """
        return self.data

    @property
    def id(self) -> str:
        """
        Id of the artifact.

        returns:
            id (str): str id of the artifact.
        """
        return f"{base64.b64encode(self.asset_path.encode())}:{self.version}"

    def get(self, name: str) -> str:
        """
        Get the artibute from the class.

        Arguments:
            name (str): name of the atribute.

        returns:
            str, byte or list of str depending on the atribute called for.
        """
        match name:
            case "type":
                return self.type
