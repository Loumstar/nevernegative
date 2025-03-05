from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel
from torch import Tensor

from nevernegative.utils.decorators import save_figure


class PlottingConfig(BaseModel):
    path: Path
    figure_size: tuple[int, int]


class Layer(ABC):
    plotting_name: str

    def __init__(self) -> None:
        self._plotting_config: PlottingConfig | None = None

        self.plot_path: Path | None = None
        self.figure_size: tuple[int, int] | None = None

    @property
    def plotting(self) -> bool:
        return self.plot_path is not None

    @contextmanager
    def setup(
        self, plot_path: Path | None = None, figure_size: tuple[int, int] | None = None
    ) -> Iterator[None]:
        try:
            self.plot_path = plot_path
            self.figure_size = figure_size

            yield

        finally:
            self.plot_path = None
            self.figure_size = None

    def _is_bw(self, image: Tensor) -> bool:
        return image.shape[-3] == 1

    def _add_image_to_axis(self, axis: Axes, image: Tensor) -> None:
        if image.ndim == 2:
            image = image.unsqueeze(-3)

        image = image.permute(1, 2, 0).clip(0, 1).cpu()
        cmap = "gray" if self._is_bw(image) else None

        axis.imshow(image, cmap=cmap)

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
        if self.plotting:
            self.plot("input.png", image)

        out = self.forward(image)

        if self.plotting:
            self.plot("output.png", out)

        return out

    @abstractmethod
    def forward(self, image: Tensor) -> Tensor:
        """Apply a transformation to the image.

        Args:
            image (Image[Any, Any]): Image to transform.

        Returns:
            Image[Any, Any]: The resultant image.
        """
