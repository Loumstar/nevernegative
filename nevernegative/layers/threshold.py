import skimage as ski

from nevernegative.layers.base import Layer
from nevernegative.typing.image import Image, ScalarTypeT, ThresholdImage


class Threshold(Layer):
    def __call__(self, image: Image[ScalarTypeT]) -> ThresholdImage:
        return image > ski.filters.threshold_mean(image)
