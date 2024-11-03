from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from nevernegative.layers.base import Layer
from nevernegative.layers.utils.decorators import save_figure


class Balancer(Layer, ABC):
    _histogram_distribution_plot_kwargs: dict[str, Any] = {}
    _histogram_cumulative_plot_kwargs: dict[str, Any] = {"linestyle": "dashed"}

    _histogram_plot_colors: tuple[str, ...] = ("red", "green", "blue")

    def __init__(
        self,
        brightness: float | tuple[float, float, float] = 0.0,
        contrast: float | tuple[float, float, float] = 0.0,
        saturation: float = 0.05,
        *,
        plot_path: Path | None = None,
        figure_size: tuple[int, int] = (15, 15),
    ) -> None:
        super().__init__(plot_path, figure_size)

        self.brightness = np.array(brightness)
        self.contrast = np.array(contrast)

        self.saturation = saturation
        self.plot_path = plot_path

    @overload
    def _histogram(
        self,
        image: NDArray,
        *,
        target_channel: int | None = None,
        return_cumulative: Literal[False] = False,
        hide_clipped_values: bool = True,
        normalize: bool = True,
        n_bins: int = 256,
    ) -> tuple[NDArray, NDArray]: ...

    @overload
    def _histogram(
        self,
        image: NDArray,
        *,
        target_channel: int | None = None,
        return_cumulative: Literal[True] = True,
        hide_clipped_values: bool = False,
        normalize: bool = True,
        n_bins: int = 256,
    ) -> tuple[NDArray, NDArray, NDArray]: ...

    def _histogram(
        self,
        image: NDArray,
        *,
        target_channel: int | None = None,
        return_cumulative: bool = False,
        hide_clipped_values: bool = False,
        normalize: bool = True,
        n_bins: int = 256,
    ) -> tuple[NDArray, NDArray] | tuple[NDArray, NDArray, NDArray]:
        clipped = np.clip(image, 0, 1)

        if hide_clipped_values:
            clipped[np.logical_or(image >= 1, image <= (1 / n_bins))] = -1

        if target_channel is not None:
            clipped = clipped[..., target_channel]

        histogram, bins = np.histogram(clipped.ravel(), bins=n_bins, range=(0, 1))

        if normalize:
            histogram = histogram / np.max(histogram)

        if not return_cumulative:
            return histogram, bins

        cumulative = np.cumsum(histogram)

        if normalize:
            cumulative /= np.max(cumulative)

        return histogram, bins, cumulative

    @save_figure
    def plot_balancing(
        self,
        image: NDArray,
        *,
        include_cumulative: bool = True,
    ) -> Figure:
        figure, axes = plt.subplots(2)
        flattened_axes: list[Axes] = axes.ravel()

        [image_axis, histogram_axis] = flattened_axes

        image_axis.imshow(image)

        for channel, color in enumerate(self._histogram_plot_colors):
            histogram, bins, cumulative = self._histogram(
                image,
                target_channel=channel,
                return_cumulative=True,
                hide_clipped_values=True,
                normalize=True,
            )

            histogram_axis.plot(
                bins[1:],
                histogram,
                color=color,
                **self._histogram_distribution_plot_kwargs,
            )

            if include_cumulative:
                histogram_axis.plot(
                    bins[:-1],
                    cumulative,
                    color=color,
                    **self._histogram_cumulative_plot_kwargs,
                )

        return figure

    def basic_balancing(self, image: NDArray) -> NDArray:
        image = ((image - 0.5) / (1 - self.contrast)) + 0.5
        image += self.brightness

        hsv = ski.color.rgb2hsv(image)
        hsv[..., 1] += self.saturation
        image = ski.color.hsv2rgb(hsv)

        return np.clip(image, 0, 1)

    @abstractmethod
    def __call__(self, image: NDArray) -> NDArray:
        """Crop an image, returning only the portion that contains the negative.

        Args:
            image (RGBImage[DTypeT]): _description_

        Returns:
            RGBImage[DTypeT]: _description_
        """
