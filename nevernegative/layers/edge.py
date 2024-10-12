import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.base import Layer


class EdgeDetect(Layer):
    def __init__(
        self,
        sigma: float,
        low_threshold: float,
        high_threshold: float,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def compute(self, image: NDArray) -> NDArray:
        return ski.feature.canny(
            image,
            sigma=self.sigma,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )
