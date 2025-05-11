from torch import Tensor

from nevernegative.layers.balancing.base import Balancer


class Invert(Balancer):
    def forward(self, image: Tensor) -> Tensor:
        return 1 - image
        # return F.invert(image)
