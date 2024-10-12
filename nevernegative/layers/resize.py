import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.base import Layer


class Resize(Layer):
    def __init__(
        self,
        height: int | None = None,
        width: int | None = None,
        ratio: float | None = None,
        anti_aliasing: bool = True,
    ) -> None:
        super().__init__()

        self.height = height
        self.width = width
        self.ratio = ratio
        self.anti_aliasing = anti_aliasing

    def _calculate_new_shape(self, image_shape: tuple[int, ...]) -> tuple[int, int]:
        if self.height is not None and self.width is not None:
            return self.height, self.width

        height, width, *_ = image_shape

        if self.ratio is not None:
            ratio = self.ratio
        elif self.height is not None:
            ratio = float(self.height) / height
        elif self.width is not None:
            ratio = float(self.width) / width
        else:
            raise RuntimeError("None of height, width or ratio have been set.")

        return (int(ratio * height), int(ratio * width))

    def compute(self, image: NDArray) -> NDArray:
        return ski.transform.resize(  # type: ignore
            image,
            self._calculate_new_shape(image.shape),
            anti_aliasing=self.anti_aliasing,
            preserve_range=True,
        )
