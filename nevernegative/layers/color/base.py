from abc import abstractmethod

from numpy.typing import NDArray

from nevernegative.layers.base import Layer


class ColorBalancer(Layer):
    @abstractmethod
    def __call__(self, image: NDArray) -> NDArray:
        """Crop an image, returning only the portion that contains the negative.

        Args:
            image (RGBImage[DTypeT]): _description_

        Returns:
            RGBImage[DTypeT]: _description_
        """
