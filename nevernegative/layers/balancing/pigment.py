from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nevernegative.layers.balancing.base import Balancer
from nevernegative.layers.balancing.presets.film.base import Film
from nevernegative.layers.common.clip import Clip
from nevernegative.layers.common.positive import Positive


class PigmentCorrection(Balancer):
    def __init__(
        self,
        film: Film,
        *,
        invert: bool = True,
        clip: bool = True,
        plot_path: Path | None = None,
        figure_size: tuple[int, int] = (15, 15),
    ) -> None:
        super().__init__(plot_path=plot_path, figure_size=figure_size)

        if film.pigment is None:
            raise NotImplementedError("Estimating pigment is not yet supported.")

        self.white = np.array(film.pigment).astype(np.float64) / 255

        self.invert = Positive(film.is_negative and invert)
        self.clip = Clip() if clip else None

    # @save_figure
    # def plot_kelvin_scale(self, image: NDArray, *, points: NDArray | None = None) -> Figure:
    #     figure, axis = plt.subplots()
    #     axis.imshow(image)

    #     if points is not None:
    #         axis.scatter(*points.T, color="green")

    #     axis.axis("off")

    #     return figure

    def __call__(self, image: NDArray) -> NDArray:
        self.plot_balancing("original.png", image)

        balanced = image / self.white

        self.plot_balancing("balanced.png", balanced)

        positive = self.invert(balanced)

        if self.clip is not None:
            positive = self.clip(positive)

        self.plot_balancing("positive.png", positive)

        return positive
