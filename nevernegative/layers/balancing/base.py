from abc import ABC

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor

from nevernegative.layers.base import Layer
from nevernegative.utils.decorators import save_figure


class Balancer(Layer, ABC):
    _color_cmap: str | None = None
    _bw_cmap: str | None = "gray"

    _color_channels: tuple[str, ...] = ("red", "green", "blue")
    _bw_channels: tuple[str, ...] = ("black",)

    def get_channel_colors(self, image: Tensor) -> tuple[str, ...]:
        match image.shape[-3]:
            case 1:
                return ("black",)
            case 3:
                return ("red", "green", "blue")
            case _:
                raise RuntimeError()

    def _add_channel_distribution_to_axis(
        self,
        axis: Axes,
        channel: Tensor,
        color: str,
        *,
        n_bins: int = 128,
        hide_clipped: bool = True,
        cumulative_axis: Axes | None = None,
    ) -> None:
        channel = channel.cpu()
        histogram, bins = torch.histogram(channel.flatten(), bins=n_bins, range=(0, 1))

        if cumulative_axis is not None:
            cumulative = torch.zeros_like(bins)
            cumulative[1:] = torch.cumsum(histogram, dim=0)

            cumulative_axis.plot(
                bins,
                cumulative,
                color=color,
                linestyle="dashed",
            )

        if hide_clipped:
            histogram[0] = 0
            histogram[-1] = 0

        axis.bar(bins[1:], histogram, width=torch.diff(bins), color=color, alpha=0.2)

    @save_figure
    def plot(self, image: Tensor, *, n_bins: int = 128, hide_clipped: bool = True) -> Figure:
        figure, axes = plt.subplots(2)
        flattened_axes: list[Axes] = axes.ravel()

        [image_axis, histogram_axis] = flattened_axes
        cumulative_axis = histogram_axis.twinx()

        self._add_image_to_axis(image_axis, image)

        for channel, color in enumerate(self.get_channel_colors(image)):
            self._add_channel_distribution_to_axis(
                histogram_axis,
                image.select(-3, channel),
                color,
                n_bins=n_bins,
                hide_clipped=hide_clipped,
                cumulative_axis=cumulative_axis,
            )

        histogram_axis.get_yaxis().set_visible(False)
        cumulative_axis.get_yaxis().set_visible(False)

        return figure
