import torchvision.transforms.functional as F
from torch import Tensor

from nevernegative.layers.base import Layer


class Blur(Layer):
    plotting_name = "blur"

    def __init__(
        self,
        kernel_size: tuple[int, int],
        sigma: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()

        self.kernel_size = list(kernel_size)
        self.sigma = list(sigma) if sigma is not None else None

    def forward(self, image: Tensor) -> Tensor:
        return F.gaussian_blur(image, self.kernel_size, self.sigma)
