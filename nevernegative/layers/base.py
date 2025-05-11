import re
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, ParamSpec, TypeVar

import kornia as K
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel
from torch import Tensor

from nevernegative.utils.decorators import save_figure

P = ParamSpec("P")
LayerT = TypeVar("LayerT", bound="Layer")


class PlotConfig(BaseModel):
    plot_path: Path
    figure_size: tuple[int, int]


class SetupConfig(BaseModel):
    image_path: Path | None
    layer_index: int

    plotting: PlotConfig | None


class Layer(ABC):
    def __init__(self) -> None:
        self._setup_config: SetupConfig | None = None

    def is_plotting(self) -> bool:
        return self.get_setup_config().plotting is not None

    def get_setup_config(self) -> SetupConfig:
        if self._setup_config is None:
            raise RuntimeError()

        return self._setup_config

    def get_layer_name(self) -> str:
        return re.sub(r"([A-Z])", r"_\1", self.__class__.__name__).strip("_").lower()

    @contextmanager
    def setup(
        self,
        image_path: Path | None,
        layer_index: int,
        *,
        plotting: PlotConfig | None = None,
    ) -> Iterator[None]:
        try:
            self._setup_config = SetupConfig(
                image_path=image_path,
                layer_index=layer_index,
                plotting=plotting,
            )

            yield

        finally:
            self._plot_config = None

    def _is_bw(self, image: Tensor) -> bool:
        return image.shape[-3] == 1

    def _add_image_to_axis(self, axis: Axes, image: Tensor) -> None:
        axis.imshow(
            K.utils.tensor_to_image(image).astype(np.float32),
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
