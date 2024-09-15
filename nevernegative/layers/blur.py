from typing import Literal

import skimage as ski

from nevernegative.layers.base import Layer
from nevernegative.typing.image import Image, ScalarTypeT


class Blur(Layer):
    def __init__(
        self,
        sigma: int | tuple[int, int],
        mode: Literal["reflect", "nearest", "mirror", "wrap"] = "nearest",
    ) -> None:
        super().__init__()

        self.sigma = (sigma, sigma) if isinstance(sigma, int) else sigma
        self.mode = mode

    def __call__(self, image: Image[ScalarTypeT]) -> Image[ScalarTypeT]:
        return ski.filters.gaussian(
            image,
            sigma=self.sigma,
            mode=self.mode,
        )
