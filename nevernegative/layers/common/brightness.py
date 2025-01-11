from typing import Any

import numpy as np
from numpy.typing import NDArray

from nevernegative.layers.base import Layer


class Brightness(Layer):
    def __init__(self, brightness: float | tuple[float, float, float]) -> None:
        self.brightness = np.array(brightness, dtype=np.float64)

    def __call__(self, image: NDArray[Any]) -> NDArray[Any]:
        return image + self.brightness
