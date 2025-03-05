from typing import Literal

import torchvision.transforms.functional as F
from torch import Tensor

from nevernegative.layers.balancing.base import Balancer


class Brightness(Balancer):
    plotting_name = "brightness"

    def __init__(
        self,
        brightness: float,
        *,
        mode: Literal["blend", "sum"] = "blend",
        channel: int | None = None,
    ) -> None:
        super().__init__()

        self.brightness = brightness

        self.mode: Literal["blend", "sum"] = mode
        self.channel = channel

    def _blend(self, image: Tensor) -> Tensor:
        return F.adjust_brightness(image, self.brightness)

    def _sum(self, image: Tensor) -> Tensor:
        return image + self.brightness

    def forward(self, image: Tensor) -> Tensor:
        function = self._blend if self.mode == "blend" else self._sum

        if self.channel is not None:
            # TODO make this non-inplace
            image[..., self.channel, :, :] = function(image.select(-3, self.channel))
            return image

        return function(image)
