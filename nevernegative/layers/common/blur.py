from typing import Literal

import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.base import Layer


class Blur(Layer):
    def __init__(
        self,
        sigma: int | tuple[int, int],
        mode: Literal["reflect", "nearest", "mirror", "wrap"] = "nearest",
    ) -> None:
        self.sigma = (sigma, sigma) if isinstance(sigma, int) else sigma
        self.mode = mode

    def __call__(self, image: NDArray) -> NDArray:
        return ski.filters.gaussian(
            image,
            sigma=self.sigma,
            mode=self.mode,
        )
