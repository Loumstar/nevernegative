from pathlib import Path

import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from nevernegative.layers.color.base import Balancer
from nevernegative.layers.common.grey import Grey
from nevernegative.layers.utils.decorators import save_figure


class WhitePointBalancer(Balancer):
    def __init__(
        self,
        brightness: float | tuple[float, float, float] = 0.0,
        contrast: float | tuple[float, float, float] = 0.0,
        saturation: float = 0.05,
        *,
        plot_path: Path | None = None,
        figure_size: tuple[int, int] = (15, 15),
        invert: bool = True,
    ) -> None:
        super().__init__(
            brightness,
            contrast,
            saturation,
            plot_path=plot_path,
            figure_size=figure_size,
        )

        self._grey = Grey()
        self._invert = invert

    @save_figure
    def plot(self, image: NDArray, *, points: NDArray | None = None) -> Figure:
        figure, axis = plt.subplots()

        if image.max() > 1:
            image /= 255

        axis.imshow(image)

        if points is not None:
            axis.scatter(*points.T, color="green")

        axis.axis("off")

        return figure

    def __call__(self, image: NDArray) -> NDArray:
        grey = self._grey(image)

        index = np.unravel_index(grey.argmax(), grey.shape)
        white = image[index]

        inverted = ski.util.invert(image) if self._invert else image

        self.plot("white.png", inverted, points=white[np.newaxis])
        self.plot_balancing("original.png", inverted)

        balanced = image.astype(np.float64) / white
        balanced = np.clip(balanced, 0, 1)

        if self._invert:
            balanced = ski.util.invert(balanced)

        self.plot_balancing("balanced.png", balanced)

        return balanced
