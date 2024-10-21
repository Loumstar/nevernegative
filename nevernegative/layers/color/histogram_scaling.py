from pathlib import Path
from typing import Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate  # type: ignore
import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.color.base import ColorBalancer


class HistogramScalingColorBalancer(ColorBalancer):
    def __init__(
        self,
        red: tuple[float, float] = (0.05, 0.95),
        green: tuple[float, float] = (0.05, 0.95),
        blue: tuple[float, float] = (0.05, 0.95),
        *,
        brightness: float | tuple[float, float, float] = 0.0,
        contrast: float | tuple[float, float, float] = 0.0,
    ) -> None:
        self.red = red
        self.blue = blue
        self.green = green

        self.brightness = np.array(brightness)
        self.contrast = np.array(contrast)

    @overload
    def _histogram(
        self,
        image: NDArray,
        *,
        channel: int | None = None,
        return_cumulative: Literal[False] = False,
        hide_clipped_values: bool = True,
        normalize: bool = True,
    ) -> tuple[NDArray, NDArray]: ...

    @overload
    def _histogram(
        self,
        image: NDArray,
        *,
        channel: int | None = None,
        return_cumulative: Literal[True] = True,
        hide_clipped_values: bool = False,
        normalize: bool = True,
    ) -> tuple[NDArray, NDArray, NDArray]: ...

    def _histogram(
        self,
        image: NDArray,
        *,
        channel: int | None = None,
        return_cumulative: bool = False,
        hide_clipped_values: bool = False,
        normalize: bool = True,
    ) -> tuple[NDArray, NDArray] | tuple[NDArray, NDArray, NDArray]:
        if hide_clipped_values:
            clipped = image.copy()
            clipped[np.logical_or(image >= 1, image <= 1 / 256)] = -1

        else:
            clipped = np.clip(image, 0, 1)

        if channel is not None:
            clipped = clipped[..., channel]

        histogram, bins = np.histogram(clipped.ravel(), bins=256, range=(0, 1))

        if normalize:
            histogram = histogram / np.max(histogram)

        if return_cumulative:
            cumulative = np.cumsum(histogram)

            if normalize:
                cumulative /= np.max(cumulative)

            return histogram, bins, cumulative

        return histogram, bins

    def plot_histogram(self, name: str, image: NDArray) -> None:
        figure, axes = plt.subplots(2)
        [image_axis, histogram_axis] = axes.ravel()

        image_axis.imshow(image)

        for channel, color in enumerate(("red", "green", "blue")):
            histogram, bins, cumulative = self._histogram(
                image,
                channel=channel,
                return_cumulative=True,
                hide_clipped_values=True,
                normalize=True,
            )

            histogram_axis.plot(bins[1:], histogram, color=color)
            histogram_axis.plot(bins[:-1], cumulative, color=color, linestyle="dashed")

        Path("results/color").mkdir(parents=True, exist_ok=True)
        figure.savefig(f"results/color/{name}.png")

    def __call__(self, image: NDArray) -> NDArray:
        inverted = ski.util.invert(image)
        self.plot_histogram("before", inverted)

        for channel, distribution_bounds in enumerate((self.red, self.green, self.blue)):
            _, bins, cumulative = self._histogram(
                inverted,
                channel=channel,
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

        inverted = ((inverted - 0.5) / (1 - self.contrast)) + 0.5
        inverted += self.brightness

        result = np.clip(inverted, 0, 1)

        self.plot_histogram("after", result)

        return result
