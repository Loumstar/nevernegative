from pathlib import Path

import numpy as np
import scipy.interpolate  # type: ignore
from numpy.typing import NDArray

from nevernegative.layers.balancing.base import Balancer
from nevernegative.layers.common.clip import Clip
from nevernegative.layers.common.positive import Positive


class HistogramScaling(Balancer):
    def __init__(
        self,
        bounds: tuple[float, float] = (0.01, 0.99),
        *,
        invert: bool = False,
        clip: bool = False,
        plot_path: Path | None = None,
        figure_size: tuple[int, int] = (15, 15),
    ) -> None:
        super().__init__(plot_path=plot_path, figure_size=figure_size)
        self.bounds = bounds

        self.invert = Positive(invert)
        self.clip = Clip() if clip else None

    def __call__(self, image: NDArray) -> NDArray:
        positive = self.invert(image)

        self.plot_balancing("original.png", positive)

        for channel in range(3):
            _, bins, cumulative = self._histogram(
                positive,
                target_channel=channel,
                return_cumulative=True,
                hide_clipped_values=False,
                normalize=True,
            )

            cumulative_to_brightness = scipy.interpolate.interp1d(
                cumulative,
                bins[:-1],
                kind="nearest",
                fill_value="extrapolate",  # type: ignore
            )

            lower, upper = np.clip(cumulative_to_brightness(self.bounds), 0, 1)
            positive[..., channel] = (positive[..., channel] - lower) / (upper - lower)

        if self.clip is not None:
            positive = self.clip(image)

        self.plot_balancing("balanced.png", positive)

        return positive
