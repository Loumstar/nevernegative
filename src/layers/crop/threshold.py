from typing import Any

import skimage as ski

from src.layers.base import Layer
from src.typing.image import GreyImage, ThresholdImage


class Threshold(Layer):
    def __call__(self, image: GreyImage[Any]) -> ThresholdImage:
        return image > ski.filters.threshold_mean(image)
