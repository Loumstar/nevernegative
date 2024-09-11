from typing import Any

import skimage as ski

from src.layers.base import Layer
from src.typing.image import DTypeT, Image


class Resize(Layer):
    def __init__(
        self,
        *,
        height: int | None = None,
        width: int | None = None,
        ratio: float | None = None,
        anti_aliasing: bool = True,
    ) -> None:
        if ratio is None and height is None and width is None:
            raise ValueError("At least one of height, width or ratio must be set.")

        elif ratio is not None and (height is not None or width is not None):
            raise ValueError("Cannot set ratio and one of height or width.")

        self.height = height
        self.width = width
        self.ratio = ratio
        self.anti_aliasing = anti_aliasing

    def _calculate_shape(self, image_shape: tuple[int, ...]) -> tuple[int, int]:
        current_height, current_width, *_ = image_shape

        if self.ratio is not None:
            return (int(current_height * self.ratio), int(current_width * self.ratio))

        if self.height is not None and self.width is not None:
            return (self.height, self.width)

        if self.height is not None:
            return (self.height, int(current_width * (self.height / current_height)))

        if self.width is None:
            raise RuntimeError("None of height, width or ratio have been set.")

        return (int(current_height * (self.width / current_width)), self.width)

    def __call__(self, image: Image[Any, DTypeT]) -> Image[Any, DTypeT]:
        resized = ski.transform.resize(
            image,
            self._calculate_shape(image.shape),
            anti_aliasing=self.anti_aliasing,
            preserve_range=True,
        )

        return resized  # type: ignore
