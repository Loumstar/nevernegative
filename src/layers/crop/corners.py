from typing import NamedTuple

import numpy as np
import skimage as ski

from src.layers.base import Layer
from src.typing.image import ColorImage, DTypeT, EdgeMap


class Line(NamedTuple):
    slope: float
    coord: tuple[float, float]
    is_vertical: bool


class CoordinateTransform(NamedTuple):
    source: tuple[int, int]
    destination: tuple[int, int]


class HoughCornerDetection(Layer):
    _START_ANGLE = np.deg2rad(-45)
    _END_ANGLE = np.deg2rad(135)
    _ANGLE_STEP = 360

    def __init__(
        self,
        *,
        threshold_ratio: float = 0.2,
        min_distance: int = 30,
        warp_correction: bool = False,
    ) -> None:
        self.threshold_ratio = threshold_ratio
        self.min_distance = min_distance
        self.warp_correction = warp_correction

        self._angles = np.linspace(
            self._START_ANGLE,
            self._END_ANGLE,
            self._ANGLE_STEP,
            endpoint=False,
        )

        if warp_correction:
            raise NotImplementedError()

    def __call__(
        self,
        image: EdgeMap,
        *,
        original_image: ColorImage[DTypeT] | None = None,
    ) -> ColorImage[DTypeT]:
        if original_image is None:
            raise RuntimeError()

        hspace, angles, distances = ski.transform.hough_line(image, theta=self._angles)

        peaks = ski.transform.hough_line_peaks(
            hspace,
            angles,
            distances,
            threshold=(self.threshold_ratio * np.max(hspace)),
            num_peaks=4,
            min_distance=self.min_distance,
        )

        lines: list[Line] = []

        # Warn if number of peaks is less than 4.
        for _, angle, distance in zip(*peaks):
            lines.append(
                Line(
                    slope=np.tan(angle + np.pi / 2),
                    coord=distance * np.array([np.cos(angle), np.sin(angle)]),
                    is_vertical=round(2 * angle / np.pi) == 0,
                )
            )

        return original_image
