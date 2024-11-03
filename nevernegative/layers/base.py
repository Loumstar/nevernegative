from abc import ABC, abstractmethod
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from nevernegative.layers.utils.decorators import save_figure


class Layer(ABC):
    def __init__(
        self,
        plot_path: Path | None = None,
        figure_size: tuple[int, int] = (15, 15),
    ) -> None:
        self.plot_path = plot_path
        self.figure_size = figure_size

    @save_figure
    def plot(self, image: NDArray) -> Figure:
        figure, axis = plt.subplots()

        if image.max() > 1:
            image /= 255

        axis.imshow(image)
        axis.axis("off")

        return figure

    @abstractmethod
    def __call__(self, image: NDArray) -> NDArray:
        """Apply a transformation to the image.

        Args:
            image (Image[Any, Any]): Image to transform.

        Returns:
            Image[Any, Any]: The resultant image.
        """
