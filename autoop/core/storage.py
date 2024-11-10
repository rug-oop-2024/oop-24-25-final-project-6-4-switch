from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """Error to inform the user that they have inputted an invalid path."""
    def __init__(self, path: str) -> None:
        """Initialize the not found error.

        Args:
            path (str): The path that causes the error.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """Abstract class for storage handling classes."""

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
    """Class that keeps track and modifies everything in the local storage."""

    def __init__(self, base_path: str = "./assets") -> None:
        """Initialize the local storage class.

        Args:
            base_path (str, optional): The directory that all the data that
            local storage saves should be. Defaults to "./assets".
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """Save the data into a folder given by key.

        Args:
            data (bytes): The data in byte that is to be saved into the file
            pointed at by key.
            key (str): The key that points to the file within base path.
        """
        path = self._join_path(key)
        # Ensure parent directories are created
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(path)
        print(data)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """Open and read the file given by key input.

        Args:
            key (str): the key that points to the file that needs to be read.

        Returns:
            bytes: the byte info of the read file.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """deletes the file that the key points to.

        Args:
            key (str, optional): The path to the file to be deleted.
            Defaults to "/".
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """Get list of all files names in prefix.

        Args:
            prefix (str, optional): prefix to folder for it to look in.
            Defaults to "/".

        Returns:
            List[str]: list of all files in the directory.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path) for p
                in keys if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None:
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        # Ensure paths are OS-agnostic
        return os.path.normpath(os.path.join(self._base_path, path))
