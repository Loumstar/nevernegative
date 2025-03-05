import torch
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from torch import Tensor

from nevernegative.layers.base import Layer


class Threshold(Layer):
    plotting_name = "threshold"

    def __init__(self, proportion: float = 1, fill_holes: bool = False) -> None:
        super().__init__()

        self.proportion = proportion
        self.fill_holes = fill_holes

    def forward(self, image: Tensor) -> Tensor:
        threshold = image > (self.proportion * image.mean())

        if self.fill_holes:
            # TODO vectorise this algorithm
            numpy_threshold = binary_fill_holes(threshold.cpu().numpy())
            numpy_threshold = remove_small_objects(numpy_threshold, connectivity=2)

            threshold = torch.tensor(numpy_threshold, dtype=torch.float32, device=image.device)

        return threshold
