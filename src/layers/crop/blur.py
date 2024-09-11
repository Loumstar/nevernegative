from typing import Literal

import skimage as ski

from src.layers.base import Layer
from src.typing.image import FloatingT, GreyImage


class GaussianBlur(Layer):
    def __init__(
        self,
        sigma: float | tuple[float, float],
        *,
        mode: Literal["reflect", "nearest", "mirror", "wrap"] = "nearest",
    ) -> None:
        self.sigma = sigma if isinstance(sigma, tuple) else (sigma, sigma)
        self.mode = mode

    def __call__(self, image: GreyImage[FloatingT]) -> GreyImage[FloatingT]:
        return ski.filters.gaussian(image, sigma=self.sigma, mode=self.mode)
