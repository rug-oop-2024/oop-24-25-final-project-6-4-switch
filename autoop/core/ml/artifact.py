from pydantic import BaseModel, Field
import base64
import platform


class Artifact(BaseModel):
    """
    Artifact.

    Attributes
    ----------
    name : str
        Name of the artifact.
    asset_path : str
        Path of the artifact.
    data : bytes
        base64 encoded path to the data.
    version : str
        Version of the artifact.
    type : str
        Type of artifact.
    metadata : dict
        Dictionary with experiment and run id.
    tags : list
        List of tags for the artifact.
    """
    name: str = Field(default=None)
    asset_path: str = Field(default=None)
    version: str = Field(default="1.0.0")
    data: bytes = Field(default=None)
    metadata: dict = Field(default={"experiment_id": None,
                                    "run_id": None})
    type: str = Field(default=None)
    tags: list[str] = Field(default=[])

    @property
    def id(self) -> str:
        """
        Return ID of the artifact.

        Returns
        -------
        Str
            ID of the artifact.
        """
        # To be OS-friendly, our id replaces the colon (:)
        # with an underscore (_).
        if platform.system() == 'Linux':
            return (f"{base64.b64encode(self.asset_path.encode())}:" +
                    f"{self.version}")

        return f"{base64.b64encode(self.asset_path.encode())}_{self.version}"

    def read(self) -> bytes:
        """
        Return data saved in this artifact.

        Returns
        -------
        None
        """
        return self.data

    def __str__(self) -> str:
        """
        Returns String representation of the artifact class.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Name of the artifact.
        """
        return self.name
