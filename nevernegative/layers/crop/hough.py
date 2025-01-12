from pathlib import Path
from typing import Literal

import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from nevernegative.layers.chain import LayerChain
from nevernegative.layers.common.edge import EdgeDetect
from nevernegative.layers.common.grey import Grey
from nevernegative.layers.common.resize import Resize
from nevernegative.layers.common.threshold import Threshold
from nevernegative.layers.crop.base import Cropper
from nevernegative.layers.utils.decorators import save_figure
from nevernegative.layers.utils.hough import HoughTransform


class HoughCrop(Cropper):
    def __init__(
        self,
        *,
        peak_ratio: float = 0.2,
        min_distance: int = 30,
        snap_to_edge_map: bool = True,
        padding: float = 0.0,
        resize: int = 800,
        edge_sigma: float = 1.0,
        edge_low_threshold: float | None = None,
        edge_high_threshold: float | None = None,
        start_angle: float = np.deg2rad(-45),
        end_angle: float = np.deg2rad(135),
        step: int = 360,
        offset: int | tuple[int, int] = 15,
        plot_path: Path | None = None,
        figure_size: tuple[int, int] = (15, 15),
    ) -> None:
        super().__init__(plot_path, figure_size)

        self.peak_ratio = peak_ratio
        self.min_distance = min_distance
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.step = step

        self.snap_to_edge_map = snap_to_edge_map
        self.padding = padding

        self.preprocess = LayerChain(
            (
                Resize(height=resize),
                Grey(),
                Threshold(),
                EdgeDetect(
                    sigma=edge_sigma,
                    low_threshold=edge_low_threshold,
                    high_threshold=edge_high_threshold,
                ),
            )
        )

        self.offset = offset

    @save_figure
    def plot(
        self,
        image: NDArray,
        *,
        lines: NDArray | None = None,
        points: NDArray | None = None,
    ) -> Figure:
        figure, axis = plt.subplots()

        if lines is not None:
            for [x, y, slope] in lines:
                axis.axline((x, y), slope=slope, color="red")

        if points is not None:
            axis.scatter(*points.T, color="green")

        axis.imshow(image)
        axis.axis("off")

        return figure

    def _image_corners(self, image: NDArray) -> NDArray[np.intp]:
        y, x, *_ = image.shape
        return np.array([(0, 0), (x, 0), (0, y), (x, y)])

    @staticmethod
    def _crop_shape(corners: NDArray) -> tuple[int, int]:
        [x, y] = np.max(corners, axis=0) - np.min(corners, axis=0)
        return int(y), int(x)

    @staticmethod
    def _image_scale(
        input: NDArray,
        output: NDArray,
    ) -> NDArray:
        input_height, input_width, *_ = input.shape
        output_height, output_width, *_ = output.shape

        return np.array([output_height / input_height, output_width / input_width])

    @staticmethod
    def _sort_coordinates(coordinates: NDArray) -> NDArray:
        if coordinates.shape[0] != 4:
            raise ValueError()

        center: NDArray = coordinates.mean(axis=0)
        vectors = coordinates - center

        angles = np.arctan2(*vectors.T)
        order = np.argsort(angles)

        return coordinates[order]

    def _get_image_center(
        self,
        image: NDArray,
        *,
        format: Literal["cartesian", "image"] = "cartesian",
    ) -> NDArray:
        center = np.array(image.shape[:2]) / 2

        if format == "cartesian":
            center = np.flip(center)

        return center

    def _apply_offset(self, coordinates: NDArray, image: NDArray) -> NDArray:
        cx, cy = self._get_image_center(image, format="cartesian")

        if isinstance(self.offset, tuple):
            dx, dy = self.offset
        else:
            dx = dy = self.offset

        coordinates[coordinates[:, 0] < cx, 0] += dx
        coordinates[coordinates[:, 0] > cx, 0] -= dx

        coordinates[coordinates[:, 1] < cy, 1] += dy
        coordinates[coordinates[:, 1] > cy, 1] -= dy

        return coordinates

    def __call__(self, image: NDArray) -> NDArray:
        edge_map = self.preprocess(image)

        hough_transform = HoughTransform(
            edge_map,
            peak_ratio=self.peak_ratio,
            min_distance=self.min_distance,
            start_angle=self.start_angle,
            end_angle=self.end_angle,
            step=self.step,
            max_num_peaks=4,
            snap_corners_to_edge_map=self.snap_to_edge_map,
            padding=self.padding,
        )

        self.plot(
            "hough.png",
            edge_map,
            lines=hough_transform.lines,
            points=hough_transform.corners,
        )

        corners = self._sort_coordinates(hough_transform.corners).astype(np.float64)
        corners = self._apply_offset(corners, edge_map)

        corners *= self._image_scale(edge_map, image)

        image_corners = self._image_corners(image)
        image_corners = self._sort_coordinates(image_corners)

        perspective_transform = ski.transform.ProjectiveTransform()

        is_success = perspective_transform.estimate(
            src=corners.astype(np.float64),
            dst=image_corners.astype(np.float64),
        )

        if not is_success:
            raise RuntimeError()

        warped = ski.transform.warp(image, perspective_transform.inverse)
        self.plot("transformed.png", warped)

        shape = self._crop_shape(corners)

        return ski.transform.resize(warped, shape)
