from typing import Any

import numpy as np
from numpy.typing import NDArray

from nevernegative.layers.base import Layer


class Clip(Layer):
    def __init__(self, lower: float = 0, upper: float = 1) -> None:
        self.lower = lower
        self.upper = upper

    def __call__(self, image: NDArray[Any]) -> NDArray[Any]:
        return np.clip(image, self.lower, self.upper)
