from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """Class for raising error for when a file is not found."""
    def __init__(self, path: str) -> None:
        """
        Initialize the NotFoundError.

        Argument:
            path (str): the path that you want to high light in the error.

        Returns:
            None
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """Base class for storage classes."""

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """LocalStorage class for managing ocal storage of this program"""

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initialize the local storage class.

        arguments
            base_path (str): Set the path of where the local storage is.

        return:
            None
        """
        self._base_path = base_path
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save file of given key.

        arguments
            key (str): relative path to the file to be saved.

        return:
            None
        """
        path = self._join_path(key)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Read file of given key.

        arguments
            key (str): relative path to the file to be read.

        return:
            bytes of the file being read.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete file of given key.

        arguments
            key (str): relative path to the file to be deleted.

        return:
            None
        """
        self._assert_path_exists(self._join_path(key))
        path = self._join_path(key)
        os.remove(path)

    def list(self, prefix: str) -> List[str]:
        """
        Get file paths of files in the specified folder and return their names.

        arguments:
            prefix (str): name of the folder to look for.

        returns
            list of file paths.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(path + "/**/*", recursive=True)
        return list(filter(os.path.isfile, keys))

    def _assert_path_exists(self, path: str) -> None:
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        return os.path.join(self._base_path, path)
