from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, overload

from numpy.typing import NDArray

from nevernegative.layers.color.base import Balancer
from nevernegative.layers.crop.base import Cropper
from nevernegative.layers.dewarp.base import Dewarper


class Scanner(ABC):
    def __init__(
        self,
        *,
        dewarper: Dewarper | None = None,
        cropper: Cropper | None = None,
        color_balancer: Balancer | None = None,
    ) -> None:
        self.dewarper = dewarper
        self.cropper = cropper
        self.color_balancer = color_balancer

    @overload
    @abstractmethod
    def file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        return_array: Literal[True],
        is_raw: bool = False,
    ) -> NDArray: ...

    @overload
    @abstractmethod
    def file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        return_array: Literal[False],
        is_raw: bool = False,
    ) -> None: ...

    @abstractmethod
    def file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        return_array: bool = False,
        is_raw: bool = False,
    ) -> NDArray | None:
        """Transform the image from a file.

        Args:
            source (str | Path): _description_
            destination (str | Path): _description_
            return_array (bool, optional): _description_. Defaults to False.
            is_raw (bool, optional): _description_. Defaults to False.
            target_layer (str | int | None, optional): _description_. Defaults to None.

        Returns:
            NDArray[Any] | None: _description_
        """

    @abstractmethod
    def array(
        self,
        image: NDArray,
    ) -> NDArray:
        """Transform the image from a numpy array.

        Args:
            image (NDArray): _description_

        Returns:
            NDArray: _description_
        """
