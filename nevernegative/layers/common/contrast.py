from typing import Any

import numpy as np
from numpy.typing import NDArray

from nevernegative.layers.base import Layer


class Contrast(Layer):
    def __init__(self, contrast: float | tuple[float, float, float]) -> None:
        self.contrast = np.array(contrast, dtype=np.float64)

    def __call__(self, image: NDArray[Any]) -> NDArray[Any]:
        return ((image - 0.5) / (1 - self.contrast)) + 0.5
