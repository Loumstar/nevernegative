from typing import Literal

import numpy as np
from numpy.typing import NDArray

from nevernegative.layers import utils


def corner_pairs(corners: NDArray) -> NDArray:
    vectors = corners[:, np.newaxis] - corners  # NxNx2
    distances = np.linalg.norm(vectors, axis=2)  # NxX
    distances[distances == 0] = np.inf

    neighbour_1, neighbour_2 = np.argsort(distances, axis=1).T[:2]  # each N
    corner_indices = np.arange(corners.shape[0])  # N

    neighbour_1_indices = np.stack((corner_indices, neighbour_1), axis=-1)  # Nx2
    neighbour_2_indices = np.stack((corner_indices, neighbour_2), axis=-1)  # Nx2

    pair_indices = np.concatenate((neighbour_1_indices, neighbour_2_indices), axis=0)  # 2Nx2
    pair_indices = np.unique(np.sort(pair_indices, axis=1), axis=0)  # Mx2

    return corners[pair_indices]


def sample_bounding_box(
    corner_pairs: NDArray,
    num_points: int,
    *,
    mode: Literal["radial", "linear"] = "linear",
    image_center: NDArray | None = None,
) -> NDArray:
    source, destination = corner_pairs.transpose((1, 0, 2))  # each Mx2

    if mode == "linear":
        t = np.linspace(0, 1, num=num_points // corner_pairs.shape[0], endpoint=False)
        samples = source[..., np.newaxis] + (destination - source)[..., np.newaxis] * t

        return np.swapaxes(samples, 1, 2).reshape(-1, 2)

    if image_center is None:
        raise ValueError("Image center must supplied when using radial.")

    vector = destination - source  # Mx2
    slope = vector[:, 1] / vector[:, 0]  # M

    corner_lines = np.concatenate((source, slope[:, np.newaxis]), axis=1)  # Mx3

    angles = np.tan(np.linspace(0, np.pi, num=num_points // 2, endpoint=False))
    radial_source = np.tile(image_center, reps=(num_points // 2, 1))
    radial_lines = np.concatenate((radial_source, angles[:, np.newaxis]), axis=1)  # Mx3

    corner_line_indices = np.arange(corner_lines.shape[0])
    radial_line_indices = np.arange(radial_lines.shape[0])

    # Create combinations of line indices and filter combinations where it is the same line.
    combinations = np.array(np.meshgrid(corner_line_indices, radial_line_indices))
    combinations = combinations.T.reshape(-1, 2)

    line_pairs = np.stack(
        (corner_lines[combinations[:, 0]], radial_lines[combinations[:, 1]]),
        axis=1,
    )

    bounds = corner_pairs[combinations[:, 0]]

    points = utils.line.intersect(line_pairs, bounds=bounds)
    mask = np.logical_or(np.isnan(points[:, 0]), np.isnan(points[:, 1]))
    points = points[~mask]

    return points
