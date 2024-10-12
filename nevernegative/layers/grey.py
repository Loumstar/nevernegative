import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.base import Layer


class Grey(Layer):
    def compute(self, image: NDArray) -> NDArray:
        if image.ndim == 2:
            return image

        if image.ndim != 3 or image.shape[-1] not in {3, 4}:
            raise RuntimeError()

        if image.shape[-1] == 4:
            image = ski.color.rgba2rgb(image)

        return ski.color.rgb2gray(image)  # type: ignore
