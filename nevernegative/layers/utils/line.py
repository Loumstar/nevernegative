from typing import NamedTuple, TypeAlias

import numpy as np

Point: TypeAlias = tuple[float, float]


class Line(NamedTuple):
    slope: float
    coord: Point


def line_intersection(line_1: Line, line_2: Line) -> Point:
    x1, y1 = line_1.coord
    x2, y2 = line_2.coord

    a = y1 - (line_1.slope * x1)
    b = y2 - (line_2.slope * x2)

    x = (b - a) / (line_1.slope - line_2.slope)

    if not np.isclose(x, x1):
        y = (line_1.slope * (x - x1)) + y1
    else:
        y = (line_2.slope * (x - x2)) + y2

    return x, y
