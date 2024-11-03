from pathlib import Path

import numpy as np
import scipy.interpolate  # type: ignore
import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.color.base import Balancer


class HistogramBalancer(Balancer):
    def __init__(
        self,
        red: tuple[float, float] = (0.05, 0.95),
        green: tuple[float, float] = (0.05, 0.95),
        blue: tuple[float, float] = (0.05, 0.95),
        *,
        brightness: float | tuple[float, float, float] = 0.0,
        contrast: float | tuple[float, float, float] = 0.0,
        saturation: float = 0.05,
        plot_path: Path | None = None,
        figure_size: tuple[int, int] = (15, 15),
    ) -> None:
        super().__init__(
            brightness,
            contrast,
            saturation,
            plot_path=plot_path,
            figure_size=figure_size,
        )

        self.red = red
        self.blue = blue
        self.green = green

    def __call__(self, image: NDArray) -> NDArray:
        inverted = ski.util.invert(image)
        self.plot_balancing("before.png", inverted)

        for channel, distribution_bounds in enumerate((self.red, self.green, self.blue)):
            _, bins, cumulative = self._histogram(
                inverted,
                target_channel=channel,
                return_cumulative=True,
                hide_clipped_values=False,
                normalize=True,
            )

            distribution_to_brightness = scipy.interpolate.interp1d(
                cumulative,
                bins[:-1],
                kind="nearest",
                fill_value="extrapolate",  # type: ignore
            )

            brightness_bounds = distribution_to_brightness(distribution_bounds)
            [lower, upper] = np.clip(brightness_bounds, 0, 1)

            inverted[..., channel] = (inverted[..., channel] - lower) / (upper - lower)

        result = self.basic_balancing(inverted)
        self.plot_balancing("after.png", result)

        return result
