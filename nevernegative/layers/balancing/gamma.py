import torchvision.transforms.functional as F
from torch import Tensor

from nevernegative.layers.balancing.base import Balancer


class Gamma(Balancer):
    plotting_name = "gamma"

    def __init__(self, gamma: float, *, channel: int | None = None) -> None:
        super().__init__()

        self.gamma = gamma
        self.channel = channel

    def forward(self, image: Tensor) -> Tensor:
        if self.channel is not None:
            # TODO make this non-inplace
            image[..., self.channel, :, :] = F.adjust_gamma(
                image[..., self.channel, :, :],
                self.gamma,
            )
            return image.nan_to_num(0)

        return F.adjust_gamma(image, self.gamma).nan_to_num(0)
