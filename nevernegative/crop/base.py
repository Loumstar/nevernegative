from abc import ABC, abstractmethod
from typing import Sequence

from nevernegative.image.image import Image
from nevernegative.layers.base import Layer


class Cropper(ABC):
    def __init__(self, layers: Sequence[Layer] | None = None) -> None:
        self.layers = layers or []

    def __call__(self, image: Image) -> Image:
        """Crop an image, returning only the portion that contains the negative.

        Args:
            image (RGBImage[DTypeT]): _description_

        Returns:
            RGBImage[DTypeT]: _description_
        """
        for layer in self.layers:
            image = layer(image)

        return self.compute(image)

    @abstractmethod
    def compute(self, image: Image) -> Image:
        """Crop an image, returning only the portion that contains the negative.

        Args:
            image (RGBImage[DTypeT]): _description_

        Returns:
            RGBImage[DTypeT]: _description_
        """
