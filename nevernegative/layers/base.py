from abc import ABC, abstractmethod

from numpy.typing import NDArray

from nevernegative.image.image import Image


class Layer(ABC):
    def __init__(self, *, name: str | None = None) -> None:
        self.name = name or str(self)

    def __str__(self) -> str:
        return self.__class__.__name__

    def __call__(self, image: Image) -> Image:
        return Image(
            source=image.source,
            block=image.block,
            layer=self.name,
            raw=self.compute(image.raw),
        )

    @abstractmethod
    def compute(self, image: NDArray) -> NDArray:
        """Apply a transformation to the image.

        Args:
            image (Image[Any, Any]): Image to transform.

        Returns:
            Image[Any, Any]: The resultant image.
        """
