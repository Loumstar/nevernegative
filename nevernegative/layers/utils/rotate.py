import torchvision.transforms.functional as F
from torch import Tensor

from nevernegative.layers.base import Layer


class Rotate(Layer):
    def __init__(self, angle: float = 0) -> None:
        super().__init__()
        self.angle = angle

    def forward(self, image: Tensor) -> Tensor:
        return F.rotate(image, self.angle)
