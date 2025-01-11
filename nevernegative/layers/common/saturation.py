from typing import Any

import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.base import Layer


class Saturation(Layer):
    def __init__(self, saturation: float) -> None:
        self.saturation = saturation

    def __call__(self, image: NDArray[Any]) -> NDArray[Any]:
        hsv = ski.color.rgb2hsv(image)
        hsv[..., 1] += self.saturation

        return ski.color.hsv2rgb(hsv)
