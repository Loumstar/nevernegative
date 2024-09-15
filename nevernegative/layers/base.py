from abc import ABC, abstractmethod

from nevernegative.typing.image import Image


class Layer(ABC):
    @abstractmethod
    def __call__(self, image: Image) -> Image:
        """Apply a transformation to the image.

        Args:
            image (Image[Any, Any]): Image to transform.

        Returns:
            Image[Any, Any]: The resultant image.
        """
