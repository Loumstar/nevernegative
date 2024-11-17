from pathlib import Path

import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from nevernegative.layers.color.base import Balancer
from nevernegative.layers.color.presets import FilmPreset
from nevernegative.layers.utils.decorators import save_figure


class WhitePointBalancer(Balancer):
    def __init__(
        self,
        preset: FilmPreset,
        plot_path: Path | None = None,
        figure_size: tuple[int, int] = (15, 15),
    ) -> None:
        super().__init__(preset, plot_path=plot_path, figure_size=figure_size)

    @save_figure
    def plot(self, image: NDArray, *, points: NDArray | None = None) -> Figure:
        figure, axis = plt.subplots()
        axis.imshow(image)

        if points is not None:
            axis.scatter(*points.T, color="green")

        axis.axis("off")

        return figure

    def __call__(self, image: NDArray) -> NDArray:
        grey = ski.color.rgb2gray(image)

        index = np.unravel_index(grey.argmax(), grey.shape)
        white = image[index]

        unbalanced = self.apply_invert(image)

        self.plot("white.png", unbalanced, points=white[np.newaxis])
        self.plot_balancing("original.png", unbalanced)

        balanced = image / white

        balanced = self.apply_invert(image)
        balanced = self.apply_contrast(image)
        balanced = self.apply_brightness(image)
        balanced = self.apply_saturation(image)
        balanced = self.apply_clip(image)
        balanced = self.apply_monochrome(image)

        self.plot_balancing("balanced.png", balanced)

        return balanced
