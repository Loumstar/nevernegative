from itertools import combinations
from typing import NamedTuple, TypeAlias

import numpy as np
import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.utils.line import Line, line_intersection

Point: TypeAlias = tuple[float, float]


class Peak(NamedTuple):
    intensity: float
    angle: float
    distance: float


class HoughTransform:
    def __init__(
        self,
        image: NDArray[np.bool],
        *,
        max_num_peaks: int | None = None,
        peak_ratio: float = 0.2,
        min_distance: int = 30,
        start_angle: float = np.deg2rad(-45),
        end_angle: float = np.deg2rad(135),
        step: int = 360,
        cache: bool = True,
    ) -> None:
        self._image = image

        self.peak_ratio = peak_ratio
        self.min_distance = min_distance
        self.cache = cache

        accumulator, angles, distances = ski.transform.hough_line(
            image,
            theta=np.linspace(
                start=start_angle,
                stop=end_angle,
                num=step,
                endpoint=False,
            ),
        )

        self._accumulator: NDArray = accumulator
        self._angles: NDArray = angles
        self._distances: NDArray = distances

        self._threshold = self.peak_ratio * np.max(self._accumulator)
        self._num_peaks = max_num_peaks or np.inf

        self._peaks: list[Peak] | None = None
        self._lines: list[Line] | None = None
        self._corners: NDArray | None = None

    @property
    def accumulator(self) -> NDArray:
        return self._accumulator

    @property
    def angles(self) -> NDArray:
        return self._angles

    @property
    def distances(self) -> NDArray:
        return self._distances

    def peaks(self) -> list[Peak]:
        if self._peaks is None or not self.cache:
            self._peaks = [
                Peak(*values)
                for values in zip(
                    *ski.transform.hough_line_peaks(
                        self.accumulator,
                        self.angles,
                        self.distances,
                        threshold=self._threshold,
                        num_peaks=self._num_peaks,
                        min_distance=self.min_distance,
                    )
                )
            ]

        return self._peaks

    def lines(self) -> list[Line]:
        if self._lines is None or not self.cache:
            self._lines = [
                Line(
                    slope=np.tan(peak.angle + np.pi / 2),
                    coord=(
                        np.cos(peak.angle) * peak.distance,
                        np.sin(peak.angle) * peak.distance,
                    ),
                )
                for peak in self.peaks()
            ]

        return self._lines

    def _filter_out_of_bound_corners(self, corners: NDArray) -> NDArray:
        height, width, *_ = self._image.shape
        x, y = corners.T

        x_mask = np.logical_and(x > 0, x < width)
        y_mask = np.logical_and(y > 0, y < height)

        return corners[np.logical_and(x_mask, y_mask)]

    def snap_to_edge_map(self, corners: NDArray) -> NDArray:
        edge_pixels = np.flip(np.argwhere(self._image), axis=1)
        vectors = np.expand_dims(corners, axis=1) - edge_pixels
        distances = np.linalg.vector_norm(vectors, axis=2)

        return edge_pixels[np.argmin(distances, axis=1)]

    def corners(
        self, *, snap_to_edge_map: bool = True, filter_out_of_bound_corners: bool = True
    ) -> NDArray:
        if self._corners is None or not self.cache:
            corners = np.array(
                [line_intersection(a, b) for a, b in combinations(self.lines(), r=2)]
            )

            if filter_out_of_bound_corners:
                corners = self._filter_out_of_bound_corners(corners)

            if snap_to_edge_map:
                corners = self.snap_to_edge_map(corners)

            self._corners = corners.astype(np.float64)

        return self._corners
