import torchvision.transforms.functional as F
from torch import Tensor

from nevernegative.layers.balancing.base import Balancer


class Grey(Balancer):
    plotting_name = "grey"

    def __init__(self, channel: int | None = 2) -> None:
        super().__init__()
        self.channel = channel

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 3 or image.shape[-3] != 3:
            raise RuntimeError()

        if self.channel is not None:
            return image[..., self.channel, :, :].unsqueeze_(-3)

        return F.rgb_to_grayscale(image)
