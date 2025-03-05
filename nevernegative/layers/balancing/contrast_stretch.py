from typing import Literal

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor

from nevernegative.layers.balancing.base import Balancer
from nevernegative.utils.brightness import compute_image_brightness
from nevernegative.utils.decorators import save_figure


class ContrastStretch(Balancer):
    plotting_name = "contrast_stretching"

    def __init__(
        self,
        bounds: tuple[float, float] = (0.01, 0.99),
        *,
        mode: Literal["mean", "brightness"] = "brightness",
        scale_factor: float = 0.1,
        per_channel: bool = False,
    ) -> None:
        super().__init__()

        self.bounds = torch.tensor(bounds, dtype=torch.float32)
        self.mode: Literal["mean", "brightness"] = mode

        self.scale_factor = scale_factor
        self.per_channel = per_channel

    @save_figure
    def plot(
        self,
        image: Tensor,
        *,
        n_bins: int = 128,
        hide_clipped: bool = True,
        bounds: list[list[float]] | None = None,
    ) -> Figure:
        figure, axes = plt.subplots(2)
        flattened_axes: list[Axes] = axes.ravel()

        [image_axis, histogram_axis] = flattened_axes
        cumulative_axis = histogram_axis.twinx()
        bounds_axis = histogram_axis.twinx()

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

            if bounds is not None:
                bounds_axis.vlines(bounds[channel], 0, 1, colors=color)

        if bounds is not None:
            bounds_axis.hlines(self.bounds, 0, 1, colors="black")

        histogram_axis.get_yaxis().set_visible(False)

        cumulative_axis.margins(y=0)
        bounds_axis.margins(y=0)

        cumulative_axis.get_yaxis().set_visible(False)
        bounds_axis.get_yaxis().set_visible(False)

        return figure

    def forward(self, image: Tensor) -> Tensor:
        if not self.per_channel:
            brightness = compute_image_brightness(image, mode=self.mode)
        else:
            brightness = image

        resized: Tensor = F.interpolate(brightness, scale_factor=0.1)
        intensities, _ = torch.sort(resized.reshape(*resized.shape[:-2], -1), 0)

        lower, upper = torch.quantile(
            intensities.unsqueeze(0),
            self.bounds.to(intensities.device),
            dim=-1,
            keepdim=True,
        )

        if self.plotting:
            bounds = torch.stack([lower, upper], dim=0).squeeze()
            self.plot("brightness.png", brightness, bounds=bounds.T.tolist())

        lower = lower.reshape(-1, 1, 1)
        upper = upper.reshape(-1, 1, 1)

        balanced = (image - lower) / (upper - lower)

        return balanced
