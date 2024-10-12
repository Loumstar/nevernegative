from typing import TypeAlias

import numpy as np
from pydantic import BaseModel

Point: TypeAlias = tuple[float, float]


class Line(BaseModel):
    slope: float
    coord: Point
    is_vertical: bool


def line_intersection(line_1: Line, line_2: Line) -> Point:
    if not (line_1.is_vertical ^ line_2.is_vertical):
        raise RuntimeError()

    xv, yv = line_1.coord
    xh, yh = line_2.coord

    a = yv - (line_1.slope * xv)
    b = yh - (line_2.slope * xh)

    x = (b - a) / (line_1.slope - line_2.slope)

    if not np.isclose(x, xv):
        y = (line_1.slope * (x - xv)) + yv
    else:
        y = (line_2.slope * (x - xh)) + yh

    return x, y
