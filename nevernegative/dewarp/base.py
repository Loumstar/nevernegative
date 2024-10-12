from abc import ABC, abstractmethod
from typing import Sequence

from nevernegative.image.image import Image
from nevernegative.layers.base import Layer


class Dewarper(ABC):
    def __init__(self, preprocessing_layers: Sequence[Layer] | None = None) -> None:
        self.preprocessing_layers = preprocessing_layers or []

    def __call__(self, image: Image) -> Image:
        """Remove warping from an image.

        Args:
            image (RGBImage[DTypeT]): _description_

        Returns:
            RGBImage[DTypeT]: _description_
        """
        for layer in self.preprocessing_layers:
            image = layer(image)

        return self.compute(image)

    @abstractmethod
    def compute(self, image: Image) -> Image:
        """Remove warping from an image.

        Args:
            image (RGBImage[DTypeT]): _description_

        Returns:
            RGBImage[DTypeT]: _description_
        """
