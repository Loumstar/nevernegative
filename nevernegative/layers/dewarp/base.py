from abc import abstractmethod

from numpy.typing import NDArray

from nevernegative.layers.base import Layer


class Dewarper(Layer):
    @abstractmethod
    def __call__(self, image: NDArray) -> NDArray:
        """Remove warping from an image.

        Args:
            image (RGBImage[DTypeT]): _description_

        Returns:
            RGBImage[DTypeT]: _description_
        """
