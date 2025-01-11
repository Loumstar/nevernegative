from typing import Any

import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.base import Layer


class Positive(Layer):
    def __init__(self, is_negative: bool = True) -> None:
        self.is_negative = is_negative

    def __call__(self, image: NDArray[Any]) -> NDArray[Any]:
        if self.is_negative:
            image = ski.util.invert(image)

        return image
