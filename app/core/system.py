from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry():
    """Class for artifact registry."""

    def __init__(self,
                 database: Database,
                 storage: Storage) -> None:
        """
        Initialize Artifactregistry class

        Arguments:
            database (Database): Database for the registry to look for datasets
            storage (Storage): Storage for rest of the artifacts.

        Returns:
            None
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Register artifact into artifactregistry

        Arguments:
            artifact (Artifact): artifact to be registered into the registry.

        Returns:
            None
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Get list of artifacts of a certain type.

        Arguments:
            type (str): type of artifacts to get from artifactregistry.

        Returns:
            list of artifacts of the type from artifact registry.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Get artifact from registry with id.

        Arguments:
            artifact_id (str): artifact id to get from the artifactregistry.

        Returns:
            artifact from artifactregistry that matches the id.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Delete artifact from artifactregistry.

        Arguments;
            artifact_id (str): id of the artifact you want to delete.

        returns:
            None
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """Class for auto machine learning system."""
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initialize AutoMLSystem class.

        Arguments:
            storage (LocalStorage): Storage lcoaltion for AutoMLStestem.
            database (Database): Database class which manages all datasets.

        Returns:
            None
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Get the singleton instance of AutoMLSystem.

        Returns:
            The singleton instance of AutoMLSystem.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Get registry of artifacts.

        Returns:
            registry of artifacts in this auto ml system.
        """
        return self._registry
