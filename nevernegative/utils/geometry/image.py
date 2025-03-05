from typing import Literal

import numpy as np
from numpy.typing import NDArray


def get_center(image: NDArray, *, format: Literal["cartesian", "image"] = "cartesian") -> NDArray:
    center = np.array(image.shape[:2]) / 2

    if format == "cartesian":
        center = np.flip(center)

    return center


def get_lengthscale(
    format: Literal["xy", "x", "y"] = "xy",
    *,
    image: NDArray | None = None,
    image_center: NDArray | None = None,
) -> NDArray:
    if image_center is None and image is not None:
        image_center = get_center(image, format="cartesian")
    elif image_center is None:
        raise ValueError()

    match format:
        case "x":
            return image_center[0]
        case "y":
            return image_center[1]
        case "xy":
            return image_center
        case _:
            raise ValueError()
