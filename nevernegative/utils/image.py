import numpy as np
import numpy.typing as npt

from nevernegative.typing.image import Image


def approximate_image_scaling(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> tuple[float, float]:
    input_height, input_width, *_ = input_shape
    output_height, output_width, *_ = output_shape

    return (output_height / input_height, output_width / input_width)


def get_image_corners(image: Image) -> npt.NDArray[np.intp]:
    return np.array(
        [
            (0, 0),
            (0, image.shape[1]),
            (image.shape[0], 0),
            (image.shape[0], image.shape[1]),
        ]
    )
