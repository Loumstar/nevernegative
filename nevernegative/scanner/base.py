from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Literal, TypeVar, overload

from nevernegative.scanner.config.base import ScannerConfig
from nevernegative.typing.image import Image, ScalarTypeT

ScannerConfigT = TypeVar("ScannerConfigT", bound=ScannerConfig)


class Scanner(ABC, Generic[ScannerConfigT]):
    def __init__(self, config: ScannerConfigT) -> None:
        self.config = config

    @overload
    @abstractmethod
    def file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        return_array: Literal[True],
        raw: bool = False,
        target_layer: str | int | None = None,
    ) -> Image[ScalarTypeT]: ...

    @overload
    @abstractmethod
    def file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        return_array: Literal[False],
        raw: bool = False,
        target_layer: str | int | None = None,
    ) -> None: ...

    @abstractmethod
    def file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        return_array: bool = False,
        raw: bool = False,
        target_layer: str | int | None = None,
    ) -> Image[ScalarTypeT] | None:
        """Transform the image from a file.

        Args:
            source (str | Path): _description_
            destination (str | Path): _description_
            return_array (bool, optional): _description_. Defaults to False.
            raw (bool, optional): _description_. Defaults to False.
            target_layer (str | int | None, optional): _description_. Defaults to None.

        Returns:
            Image[ScalarTypeT] | None: _description_
        """

    @abstractmethod
    def glob(
        self,
        source: str,
        destination: str,
        *,
        target_layer: str | int | None = None,
    ) -> None:
        """Transform images in a directory.

        Args:
            source (str): _description_
            destination (str): _description_
            target_layer (str | int | None, optional): _description_. Defaults to None.
        """

    @abstractmethod
    def array(
        self,
        source: Image[ScalarTypeT],
        *,
        target_layer: str | int | None = None,
    ) -> Image[ScalarTypeT]:
        """Transform the image from a numpy array.

        Args:
            source (Image[ScalarTypeT]): _description_

        Returns:
            Image[ScalarTypeT]: _description_
        """
