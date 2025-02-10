from abc import abstractmethod

from torch import Tensor

from nevernegative.layers.base import Layer


class Cropper(Layer):
    @abstractmethod
    def __call__(self, image: Tensor) -> Tensor:
        """Crop an image, returning only the portion that contains the negative.

        Args:
            image (RGBImage[DTypeT]): _description_

        Returns:
            RGBImage[DTypeT]: _description_
        """
