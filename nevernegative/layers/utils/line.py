import numpy as np
from numpy.typing import NDArray


def intersect(
    lines: NDArray,
    *,
    bounds: NDArray | None = None,
    eps: float = 1e-9,
) -> NDArray:
    [xs, ys, slope] = np.moveaxis(lines, -1, 0)  # Each are Nx2

    [x1, x2] = xs.T  # N
    [y1, y2] = ys.T  # N
    [slope_1, slope_2] = slope.T  # N

    a = y1 - (slope_1 * x1)
    b = y2 - (slope_2 * x2)

    # Handle divide by zero warnings
    # the intersection values will be large and eventually thrown out.
    slope_1[slope_1 == slope_2] += eps

    x = (b - a) / (slope_1 - slope_2)

    y = np.where(
        np.isclose(x, x1),
        (slope_2 * (x - x2)) + y2,
        (slope_1 * (x - x1)) + y1,
    )

    if bounds is not None:
        x_bounds, y_bounds = np.moveaxis(bounds, 2, 0)
        x_bounds = np.sort(x_bounds, axis=1)
        y_bounds = np.sort(y_bounds, axis=1)

        # If it falls outside of the bounds, set the result to NaN
        x[np.logical_or(x < x_bounds[:, 0], x > x_bounds[:, 1])] = np.nan
        y[np.logical_or(y < y_bounds[:, 0], y > y_bounds[:, 1])] = np.nan

    return np.stack((x, y), axis=-1)
