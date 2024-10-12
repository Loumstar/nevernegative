from typing import Literal

import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.base import Layer


class Blur(Layer):
    def __init__(
        self,
        name: str,
        sigma: int | tuple[int, int],
        mode: Literal["reflect", "nearest", "mirror", "wrap"] = "nearest",
    ) -> None:
        super().__init__(name=name)

        self.sigma = (sigma, sigma) if isinstance(sigma, int) else sigma
        self.mode = mode

    def compute(self, image: NDArray) -> NDArray:
        return ski.filters.gaussian(
            image,
            sigma=self.sigma,
            mode=self.mode,
        )
