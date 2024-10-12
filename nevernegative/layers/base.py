from abc import ABC, abstractmethod

from numpy.typing import NDArray


class Layer(ABC):
    @abstractmethod
    def __call__(self, image: NDArray) -> NDArray:
        """Apply a transformation to the image.

        Args:
            image (Image[Any, Any]): Image to transform.

        Returns:
            Image[Any, Any]: The resultant image.
        """
