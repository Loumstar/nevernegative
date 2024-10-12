import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.base import Layer


class Threshold(Layer):
    def __call__(self, image: NDArray) -> NDArray:
        return image > ski.filters.threshold_mean(image)
