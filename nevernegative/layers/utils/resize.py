import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.transforms import InterpolationMode

from nevernegative.layers.base import Layer


class Resize(Layer):
    plotting_name = "resize"

    def __init__(
        self,
        height: int | None = None,
        width: int | None = None,
        ratio: float | None = None,
        anti_aliasing: bool = True,
    ) -> None:
        super().__init__()

        self.height = height
        self.width = width
        self.ratio = ratio
        self.anti_aliasing = anti_aliasing

    def _calculate_shape(self, image_shape: tuple[int, ...]) -> list[int]:
        if self.height is not None and self.width is not None:
            return [self.height, self.width]

        *_, height, width = image_shape

        if self.ratio is not None:
            ratio = self.ratio
        elif self.height is not None:
            ratio = float(self.height) / height
        elif self.width is not None:
            ratio = float(self.width) / width
        else:
            raise RuntimeError("None of height, width or ratio have been set.")

        return [int(ratio * height), int(ratio * width)]

    def forward(self, image: Tensor) -> Tensor:
        return F.resize(
            image,
            self._calculate_shape(image.shape),
            interpolation=InterpolationMode.NEAREST,
            antialias=self.anti_aliasing,
        )
