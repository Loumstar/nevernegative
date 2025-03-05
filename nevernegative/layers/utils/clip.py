from torch import Tensor

from nevernegative.layers.base import Layer


class Clip(Layer):
    plotting_name = "clip"

    def __init__(self, lower: float = 0, upper: float = 1) -> None:
        super().__init__()

        self.lower = lower
        self.upper = upper

    def forward(self, image: Tensor) -> Tensor:
        return image.clip(self.lower, self.upper)
