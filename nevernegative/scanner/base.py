from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import rawpy
import skimage as ski
import torch
from torch import Tensor

from nevernegative.layers.typing import LayerCallable


class Scanner(ABC):
    def __init__(self, layers: Sequence[LayerCallable]) -> None:
        self.layers = layers

    def _from_file(self, source: str | Path, *, is_raw: bool = False) -> Tensor:
        if isinstance(source, str):
            source = Path(source)

        if is_raw:
            with rawpy.imread(str(source)) as raw:
                image = raw.postprocess().copy()
        else:
            image = ski.io.imread(source)

        return torch.tensor(image, dtype=torch.float64)

    @abstractmethod
    def file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        is_raw: bool = False,
    ) -> Tensor:
        """Transform the image from a file.

        Args:
            source (str | Path): _description_
            destination (str | Path): _description_
            return_array (bool, optional): _description_. Defaults to False.
            is_raw (bool, optional): _description_. Defaults to False.
            target_layer (str | int | None, optional): _description_. Defaults to None.

        Returns:
            Tensor[Any] | None: _description_
        """

    @abstractmethod
    def array(
        self,
        image: Tensor,
    ) -> Tensor:
        """Transform the image from a numpy array.

        Args:
            image (Tensor): _description_

        Returns:
            Tensor: _description_
        """
