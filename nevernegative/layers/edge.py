from typing import Any

import skimage as ski

from nevernegative.layers.base import Layer
from nevernegative.typing.image import EdgeMap, Image


class EdgeDetect(Layer):
    def __init__(
        self,
        sigma: float,
        low_threshold: float,
        high_threshold: float,
    ) -> None:
        super().__init__()

        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, image: Image[Any]) -> EdgeMap:
        return ski.feature.canny(
            image,
            sigma=self.sigma,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )
