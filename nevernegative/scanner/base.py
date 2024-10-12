from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Literal, TypeVar, overload

import rawpy
import skimage as ski
from numpy.typing import NDArray

from nevernegative.scanner.config.base import ScannerConfig

ScannerConfigT = TypeVar("ScannerConfigT", bound=ScannerConfig)


class Scanner(ABC, Generic[ScannerConfigT]):
    def __init__(self, config: ScannerConfigT) -> None:
        self.config = config

    def from_file(self, source: str | Path, *, is_raw: bool = False) -> NDArray:
        if isinstance(source, str):
            source = Path(source)

        if is_raw:
            with rawpy.imread(source) as raw:
                image = raw.postprocess().copy()
        else:
            image = ski.io.imread(source)

        return image

    @overload
    @abstractmethod
    def file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        return_array: Literal[True],
        raw: bool = False,
    ) -> NDArray: ...

    @overload
    @abstractmethod
    def file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        return_array: Literal[False],
        raw: bool = False,
    ) -> None: ...

    @abstractmethod
    def file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        return_array: bool = False,
        raw: bool = False,
    ) -> NDArray | None:
        """Transform the image from a file.

        Args:
            source (str | Path): _description_
            destination (str | Path): _description_
            return_array (bool, optional): _description_. Defaults to False.
            raw (bool, optional): _description_. Defaults to False.
            target_layer (str | int | None, optional): _description_. Defaults to None.

        Returns:
            NDArray[Any] | None: _description_
        """

    @abstractmethod
    def glob(
        self,
        source: str,
        destination: str,
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
        source: NDArray,
    ) -> NDArray:
        """Transform the image from a numpy array.

        Args:
            source (NDArray): _description_

        Returns:
            NDArray: _description_
        """
