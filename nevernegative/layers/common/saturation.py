import warnings
from typing import Any

import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.base import Layer


class Saturation(Layer):
    def __init__(self, saturation: float) -> None:
        self.saturation = saturation

    def __call__(self, image: NDArray[Any]) -> NDArray[Any]:
        # Numpy bug in rgb2hsv raises warnings that should be silenced.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            hsv = ski.color.rgb2hsv(image)
            hsv[..., 1] += self.saturation

            return ski.color.hsv2rgb(hsv)
