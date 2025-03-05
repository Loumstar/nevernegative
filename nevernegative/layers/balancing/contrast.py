import torchvision.transforms.functional as F
from torch import Tensor

from nevernegative.layers.balancing.base import Balancer


class Contrast(Balancer):
    plotting_name = "contrast"

    def __init__(self, contrast: float, *, channel: int | None = None) -> None:
        super().__init__()

        self.contrast = contrast
        self.channel = channel

    def forward(self, image: Tensor) -> Tensor:
        if self.channel is not None:
            # TODO make this non-inplace
            image[..., self.channel, :, :] = F.adjust_contrast(
                image[..., self.channel, :, :].unsqueeze(-3),
                self.contrast,
            ).squeeze(-3)

            return image

        return F.adjust_contrast(image, self.contrast)
