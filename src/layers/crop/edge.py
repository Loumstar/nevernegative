from typing import Any

import skimage as ski

from src.layers.base import Layer
from src.typing.image import EdgeMap, GreyImage


class CannyEdge(Layer):
    def __call__(self, image: GreyImage[Any]) -> EdgeMap:
        return ski.feature.canny(image)
