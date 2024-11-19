from pathlib import Path

import numpy as np
import scipy.interpolate  # type: ignore
from numpy.typing import NDArray

from nevernegative.layers.color.base import Balancer
from nevernegative.layers.color.presets import FilmPreset


class HistogramBalancer(Balancer):
    def __init__(
        self,
        preset: FilmPreset,
        *,
        plot_path: Path | None = None,
        figure_size: tuple[int, int] = (15, 15),
    ) -> None:
        super().__init__(preset, plot_path=plot_path, figure_size=figure_size)

    def __call__(self, image: NDArray) -> NDArray:
        image = self.apply_invert(image)
        self.plot_balancing("original.png", image)

        for channel_index, channel in enumerate(self.preset.iter_channels()):
            _, bins, cumulative = self._histogram(
                image,
                target_channel=channel_index,
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

            lower, upper = np.clip(cumulative_to_brightness(channel.bounds), 0, 1)
            image[..., channel_index] = (image[..., channel_index] - lower) / (upper - lower)

        image = self.apply_contrast(image)
        image = self.apply_brightness(image)
        image = self.apply_saturation(image)
        image = self.apply_monochrome(image)

        image = self.apply_clip(image)

        self.plot_balancing("balanced.png", image)

        return image
