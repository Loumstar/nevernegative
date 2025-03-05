import torch
import torch.linalg
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from torch import Tensor

from nevernegative.layers.balancing.grey import Grey
from nevernegative.layers.crop.base import Cropper
from nevernegative.layers.utils.chain import Chain
from nevernegative.layers.utils.resize import Resize
from nevernegative.layers.utils.threshold import Threshold
from nevernegative.utils.decorators import save_figure


class BoxCrop(Cropper):
    plotting_name = "crop"

    def __init__(
        self,
        *,
        resize: int = 800,
        threshold: float = 1,
        padding: float | tuple[float, float] = 0,
        fill_holes: bool = False,
    ) -> None:
        super().__init__()

        self.preprocess = Chain(
            (
                Resize(height=resize),
                Grey(),
                Threshold(threshold),
            )
        )

        self.padding = torch.tensor([padding], dtype=torch.float32).squeeze()
        self.fill_holes = fill_holes

    @save_figure
    def plot(
        self,
        image: Tensor,
        *,
        points: Tensor | None = None,
    ) -> Figure:
        figure, axis = plt.subplots()

        self._add_image_to_axis(axis, image)

        if points is not None:
            axis.scatter(*points.T.tolist(), color="green")

        return figure

    def _add_padding(self, corners: Tensor) -> Tensor:
        centroid = corners.mean(0)
        padding = self.padding.to(corners.device)

        vector = corners - centroid

        return (vector * (padding + 1)) + centroid

    def forward(self, image: Tensor) -> Tensor:
        threshold = self.preprocess(image).squeeze(-3)

        if self.fill_holes:
            # TODO vectorise this algorithm
            numpy_threshold = binary_fill_holes(threshold.cpu().numpy())
            numpy_threshold = remove_small_objects(numpy_threshold, connectivity=2)

            threshold = torch.tensor(numpy_threshold, dtype=torch.float32, device=image.device)

        coords = threshold.cpu().nonzero().to(torch.float32).to(image.device)
        corners = self._get_corners(coords, threshold.shape)

        corners = corners.flip(-1)

        if self.padding:
            corners = self._add_padding(corners)

        if self.plotting:
            self.plot("threshold.png", threshold)

        ratio = torch.tensor(image.shape[-2:]) / torch.tensor(threshold.shape[-2:])
        corners *= ratio.to(image.device)

        if self.plotting:
            self.plot("corners.png", image, points=corners)

        *_, h, w = image.shape

        return F.perspective(
            image,
            startpoints=corners.tolist(),
            endpoints=[[0, 0], [w, 0], [w, h], [0, h]],
        )
