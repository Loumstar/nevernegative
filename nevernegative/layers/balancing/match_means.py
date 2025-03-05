import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor

from nevernegative.layers.balancing.base import Balancer
from nevernegative.utils.decorators import save_figure


class MatchMeans(Balancer):
    plotting_name = "match_means"

    def __init__(
        self, shift_channel: int, target_channel: int, *, steps: int = 512, bins: int = 512
    ) -> None:
        super().__init__()

        self.shift_channel = shift_channel
        self.target_channel = target_channel

        self.steps = steps
        self.bins = bins

        self.bounds = torch.tensor([0.5], dtype=torch.float32)

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
        resized: Tensor = F.interpolate(image, scale_factor=0.1)
        intensities, _ = torch.sort(resized.reshape(*resized.shape[:-2], -1), 0)

        pivots = torch.quantile(
            intensities.unsqueeze(0),
            self.bounds.to(intensities.device),
            dim=-1,
            keepdim=True,
        )

        pivots.squeeze_()

        if self.plotting:
            self.plot("brightness.png", image, bounds=pivots.tolist())

        shift = pivots[self.target_channel] - pivots[self.shift_channel]

        balanced = image.detach().clone()
        balanced[..., self.shift_channel, :, :] += shift

        return balanced
