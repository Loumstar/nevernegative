from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, ParamSpec, TypeVar

import kornia as K
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel
from torch import Tensor

from nevernegative.utils.decorators import save_figure

P = ParamSpec("P")
LayerT = TypeVar("LayerT", bound="Layer")


class DebugConfig(BaseModel):
    plot_path: Path
    figure_size: tuple[int, int]


class Layer(ABC):
    plotting_name: str

    def __init__(self) -> None:
        self._debug_config: DebugConfig | None = None

        self._image_path: Path | None = None
        self._layer_index: int | None = None

    @contextmanager
    def setup(
        self,
        image_path: Path | None,
        layer_index: int,
        *,
        debug: DebugConfig | None = None,
    ) -> Iterator[None]:
        try:
            self._image_path = image_path
            self._layer_index = layer_index
            self._debug_config = debug

            yield

        finally:
            self._debug_config = None

    def _is_bw(self, image: Tensor) -> bool:
        return image.shape[-3] == 1

    def _add_image_to_axis(self, axis: Axes, image: Tensor) -> None:
        axis.imshow(
            K.utils.tensor_to_image(image),
            cmap="gray" if self._is_bw(image) else None,
        )

    @save_figure
    def plot(self, image: Tensor) -> Figure:
        figure, axis = plt.subplots()
        self._add_image_to_axis(axis, image)

        return figure

    def __call__(self, image: Tensor) -> Tensor:
        """Apply a transformation to the image.

        Args:
            image (Image[Any, Any]): Image to transform.

        Returns:
            Image[Any, Any]: The resultant image.
        """
        out = self.forward(image)
        self.plot("result.png", out)

        return out

    @abstractmethod
    def forward(self, image: Tensor) -> Tensor:
        """Apply a transformation to the image.

        Args:
            image (Image[Any, Any]): Image to transform.

        Returns:
            Image[Any, Any]: The resultant image.
        """
