import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.base import Layer


class EdgeDetect(Layer):
    def __init__(
        self,
        *,
        sigma: float = 1,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
    ) -> None:
        super().__init__()

        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, image: NDArray) -> NDArray:
        return ski.feature.canny(
            image,
            sigma=self.sigma,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )
