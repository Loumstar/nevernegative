from typing import Literal

import numpy as np
from numpy.typing import NDArray

from nevernegative.layers import utils


def snap_to_edge_map(
    points: NDArray,
    image: NDArray[np.bool],
    *,
    image_center: NDArray | None = None,
    t_weight: float = 10.0,
    method: Literal["radial", "linear"] = "linear",
) -> NDArray:
    edge_pixels = np.flip(np.argwhere(image), axis=1)  # M,2

    if method == "linear":
        vectors = np.expand_dims(points, axis=1) - edge_pixels
        distances = np.linalg.vector_norm(vectors, axis=2)

        return edge_pixels[np.argmin(distances, axis=1)]

    if image_center is None:
        image_center = utils.image.get_center(image, format="cartesian")

    vectors = image_center - points  # Nx2
    euclidean = np.linalg.norm(vectors, axis=1)  # N
    snap_vector = edge_pixels[:, np.newaxis] - points  # MxNx2

    t = np.sum(snap_vector * vectors, axis=2) / (euclidean**2)  # MxN

    projections: NDArray = (t[..., np.newaxis] * vectors) + points[: np.newaxis]
    score = np.linalg.norm(projections - edge_pixels[:, np.newaxis], axis=2)
    score += t_weight * np.abs(t)  # MxN

    return edge_pixels[np.argmin(score, axis=0)]
