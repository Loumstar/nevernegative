import numpy as np
import skimage as ski
from numpy.typing import NDArray


class HoughTransform:
    def __init__(
        self,
        image: NDArray[np.bool],
        *,
        snap_corners_to_edge_map: bool = True,
        max_num_peaks: int | None = None,
        peak_ratio: float = 0.2,
        min_distance: int = 30,
        start_angle: float = np.deg2rad(-45),
        end_angle: float = np.deg2rad(135),
        step: int = 360,
        cache: bool = True,
    ) -> None:
        self._image = image

        self.snap_corners_to_edge_map = snap_corners_to_edge_map

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

        self._peaks: NDArray | None = None
        self._lines: NDArray | None = None
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

    def _intersect(self, lines: NDArray, eps: float = 1e-6) -> NDArray:
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

        return np.stack((x, y), axis=-1)

    @property
    def peaks(self) -> NDArray:
        if self._peaks is not None and self.cache:
            return self._peaks

        accumulator, angles, distances = ski.transform.hough_line_peaks(
            self.accumulator,
            self.angles,
            self.distances,
            threshold=self._threshold,
            num_peaks=self._num_peaks,
            min_distance=self.min_distance,
        )

        self._peaks = peaks = np.stack((accumulator, angles, distances), axis=-1)

        return peaks

    @property
    def lines(self) -> NDArray:
        if self._lines is not None and self.cache:
            return self._lines

        [angles, distances] = self.peaks.T[1:]

        slopes = np.tan(angles + np.pi / 2)

        x = np.cos(angles) * distances
        y = np.sin(angles) * distances

        self._lines = lines = np.stack([x, y, slopes], axis=-1)

        return lines

    def _filter_out_of_bound_corners(self, corners: NDArray) -> NDArray:
        height, width, *_ = self._image.shape
        x, y = corners.T

        x_mask = np.logical_and(x > 0, x < width)
        y_mask = np.logical_and(y > 0, y < height)

        return corners[np.logical_and(x_mask, y_mask)]

    def snap(self, corners: NDArray) -> NDArray:
        edge_pixels = np.flip(np.argwhere(self._image), axis=1)
        vectors = np.expand_dims(corners, axis=1) - edge_pixels
        distances = np.linalg.vector_norm(vectors, axis=2)

        return edge_pixels[np.argmin(distances, axis=1)]

    @property
    def corners(self) -> NDArray:
        if self._corners is not None and self.cache:
            return self._corners

        indices = np.arange(self.lines.shape[0])

        # Create combinations of line indices and filter combinations where it is the same line.
        combinations = np.array(np.meshgrid(indices, indices)).T.reshape(-1, 2)

        combinations = combinations[combinations[:, 0] != combinations[:, 1]]
        combinations = np.unique(np.sort(combinations, axis=1), axis=0)

        lines_to_intersect = self.lines[combinations].astype(np.float64)  # Nx2x3

        corners = self._intersect(lines_to_intersect)
        corners = self._filter_out_of_bound_corners(corners)

        if self.snap_corners_to_edge_map:
            corners = self.snap(corners)

        self._corners = corners

        return corners
