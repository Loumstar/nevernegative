from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from nevernegative.layers.balancing.base import Balancer
from nevernegative.layers.common.grey import Grey
from nevernegative.layers.common.positive import Positive
from nevernegative.layers.utils.decorators import save_figure


class WhiteBalance(Balancer):
    def __init__(
        self,
        *,
        invert: bool = False,
        plot_path: Path | None = None,
        figure_size: tuple[int, int] = (15, 15),
    ) -> None:
        super().__init__(plot_path=plot_path, figure_size=figure_size)

        self.make_positive = Positive(invert)
        self.grey = Grey()

    @save_figure
    def plot(self, image: NDArray, *, points: NDArray | None = None) -> Figure:
        figure, axis = plt.subplots()
        axis.imshow(image)

        if points is not None:
            axis.scatter(*points.T, color="green")

        axis.axis("off")

        return figure

    def __call__(self, image: NDArray) -> NDArray:
        positive = self.make_positive(image)

        grey = self.grey(positive)

        index = np.unravel_index(grey.argmax(), grey.shape)
        white = positive[index]

        self.plot("white.png", image, points=white[np.newaxis])
        self.plot_balancing("original.png", positive)

        balanced = positive / white

        self.plot_balancing("balanced.png", balanced)

        return balanced
