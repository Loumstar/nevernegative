from math import ceil, sqrt
from typing import Sequence

import numpy as np
import skimage as ski
import torch
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor
from typing_extensions import deprecated

from nevernegative.layers.balancing.grey import Grey
from nevernegative.layers.base import Layer
from nevernegative.layers.crop.base import Cropper
from nevernegative.layers.utils.blur import Blur
from nevernegative.layers.utils.chain import Chain
from nevernegative.layers.utils.edge import EdgeDetect
from nevernegative.layers.utils.resize import Resize
from nevernegative.layers.utils.threshold import Threshold
from nevernegative.utils.decorators import save_figure


class HoughCrop(Cropper):
    plotting_name = "crop"

    default_preprocessing_layers: Sequence[Layer] = (
        Resize(height=800),
        Grey(),
        Blur((5, 5)),
        Threshold(),
        EdgeDetect(sigma=1),
    )

    def __init__(
        self,
        padding: float | tuple[float, float] = 0,
        snap: bool = True,
        *,
        preprocessing_layers: Sequence[Layer] | None = None,
    ) -> None:
        super().__init__()

        self.snap = snap
        self.padding = torch.tensor([padding], dtype=torch.float32).squeeze()

        if preprocessing_layers is None:
            preprocessing_layers = self.default_preprocessing_layers

        self.preprocess = Chain(preprocessing_layers)

    @save_figure
    def plot(
        self,
        image: Tensor,
        *,
        lines: Tensor | None = None,
        points: Tensor | None = None,
    ) -> Figure:
        figure, axis = plt.subplots()

        self._add_image_to_axis(axis, image)

        if lines is not None:
            for [x, y, slope] in lines.tolist():
                axis.axline((x, y), slope=slope, color="red")

        if points is not None:
            axis.scatter(*points.T.tolist(), color="green")

        return figure

    @deprecated("Using skimage instead.")
    def hough_transform(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        theta = torch.linspace(np.deg2rad(-45), np.deg2rad(135), steps=180, dtype=torch.float64)

        offset = ceil(sqrt(image.shape[-2] ** 2 + image.shape[-1] ** 2))
        max_distance = 2 * offset + 1

        pixels = torch.nonzero(image).unsqueeze(-2)  # n x 1 x 2

        indices = pixels[..., 0] * theta.cos() + pixels[..., 1] * theta.sin()
        indices = (indices.round() + offset).to(torch.int64)

        accumulator_indices, accumulator_counts = indices.unique(False, True, dim=0)

        accumulator = torch.zeros((max_distance, theta.shape[0]))
        accumulator[accumulator_indices] = accumulator_counts

        bins = torch.linspace(-offset, offset, steps=max_distance)

        return accumulator, bins, theta

    def peaks(self, image: Tensor) -> tuple[Tensor, Tensor]:
        accumulator, angles, distances = ski.transform.hough_line(
            image.squeeze().cpu().numpy(),
            theta=np.linspace(
                start=np.deg2rad(-45),
                stop=np.deg2rad(135),
                num=180,
                endpoint=False,
            ),
        )

        accumulator, angles, distances = ski.transform.hough_line_peaks(
            accumulator,
            angles,
            distances,
            threshold=0.2 * accumulator.max(),
            num_peaks=4,
            min_distance=50,
        )

        return (
            torch.tensor(
                angles,
                dtype=torch.float32,
                device=image.device,
            ),
            torch.tensor(
                distances,
                dtype=torch.float32,
                device=image.device,
            ),
        )

    def intersect(self, verticals: Tensor, horizontals: Tensor, *, eps: float = 1e-9) -> Tensor:
        xv, yv, sv = verticals.unbind(-1)
        xh, yh, sh = horizontals.unbind(-1)

        a = yv - (sv * xv)
        b = yh - (sh * xh)

        x = (b - a) / ((sv - sh) + eps)

        y = torch.where(
            torch.isclose(x, xv),
            (sh * (x - xh)) + yh,
            (sv * (x - xv)) + yv,
        )

        return torch.stack((x, y), dim=-1)

    def _add_padding(self, corners: Tensor) -> Tensor:
        centroid = corners.mean(0)
        padding = self.padding.to(corners.device)

        vector = corners - centroid

        return (vector * (padding + 1)) + centroid

    def _snap_to_edge(self, corners: Tensor, edge_map: Tensor) -> Tensor:
        pixels = edge_map.squeeze().nonzero().to(corners.dtype)  # n x 2

        vectors = corners.unsqueeze(-2) - pixels
        distances: Tensor = torch.linalg.vector_norm(vectors, dim=-1)

        return pixels[distances.argmin(dim=-1)]

    def forward(self, image: Tensor) -> Tensor:
        with self.preprocess.setup(
            plot_path=self.plot_path / "chain" if self.plot_path is not None else None,
            figure_size=self.figure_size,
        ):
            edge_map = self.preprocess(image)

        angles, distances = self.peaks(edge_map)

        if angles.shape[0] < 4:
            raise RuntimeError(f"Could not find more than {angles.shape[0]} lines.")

        indices = angles.argsort()

        # Sort so we can split them by angle later
        angles = angles[indices]
        distances = distances[indices]

        slopes = (angles + (torch.pi / 2)).tan()
        x = distances * angles.cos()
        y = distances * angles.sin()

        lines = torch.stack((x, y, slopes), dim=-1)  # 4 x 3
        verticals, horizontals = lines.split(2)

        vi, hi = torch.cartesian_prod(torch.arange(2), torch.arange(2)).unbind(1)

        coords = self.intersect(verticals[vi], horizontals[hi]).flip(-1)
        corners = self._get_corners(coords, edge_map.shape)

        if self.snap:
            corners = self._snap_to_edge(corners, edge_map)

        corners = corners.flip(-1)

        if (self.padding != 0).any():
            corners = self._add_padding(corners)

        if self.plotting:
            self.plot("edge_map.png", edge_map)

        image_size = torch.tensor(image.shape[-2:])
        edge_map_size = torch.tensor(edge_map.shape[-2:])

        ratio = (image_size / edge_map_size).to(image.device)

        corners *= ratio
        lines[..., :2] *= ratio

        if self.plotting:
            self.plot("corners.png", image, points=corners, lines=lines)

        *_, h, w = image.shape

        return F.perspective(
            image,
            startpoints=corners.tolist(),
            endpoints=[[0, 0], [w, 0], [w, h], [0, h]],
        )
