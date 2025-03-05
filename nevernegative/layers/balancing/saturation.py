import torchvision.transforms.functional as F
from torch import Tensor

from nevernegative.layers.balancing.base import Balancer


class Saturation(Balancer):
    plotting_name = "saturation"

    def __init__(self, saturation: float) -> None:
        super().__init__()

        self.saturation = saturation

    def forward(self, image: Tensor) -> Tensor:
        return F.adjust_saturation(image, self.saturation)
