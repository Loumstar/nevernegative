import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor

from nevernegative.layers.balancing.base import Balancer
from nevernegative.utils.brightness import compute_image_brightness
from nevernegative.utils.decorators import save_figure


class ShiftBound(Balancer):
    plotting_name = "match_means"

    def __init__(self, bound: float, value: float) -> None:
        super().__init__()

        self.bounds = torch.tensor([bound], dtype=torch.float32)
        self.value = value

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
            bounds_axis.vlines(bounds, 0, 1, colors="black")
            bounds_axis.hlines(self.bounds, 0, 1, colors="black")

        histogram_axis.get_yaxis().set_visible(False)

        cumulative_axis.margins(y=0)
        bounds_axis.margins(y=0)

        cumulative_axis.get_yaxis().set_visible(False)
        bounds_axis.get_yaxis().set_visible(False)

        return figure

    def forward(self, image: Tensor) -> Tensor:
        brightness = compute_image_brightness(image, mode="brightness")

        resized: Tensor = F.interpolate(brightness, scale_factor=0.1)
        intensities, _ = torch.sort(resized.reshape(*resized.shape[:-2], -1), 0)

        pivots = torch.quantile(
            intensities.unsqueeze(0),
            self.bounds.to(intensities.device),
            dim=-1,
            keepdim=True,
        )

        pivot_value: float = pivots.squeeze().tolist()  # type: ignore

        if self.plotting:
            self.plot("brightness.png", image, bounds=[[pivot_value]])

        shift = self.value - pivot_value

        return image + shift
