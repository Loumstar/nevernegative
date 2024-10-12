import itertools

import numpy as np
import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.common.config.edge import EdgeDetectConfig
from nevernegative.layers.common.config.threshold import ThresholdConfig
from nevernegative.layers.common.edge import EdgeDetect
from nevernegative.layers.common.grey import Grey
from nevernegative.layers.common.threshold import Threshold
from nevernegative.layers.config import LayerConfig
from nevernegative.layers.crop.base import Cropper
from nevernegative.layers.crop.config.hough import HoughTransformParameters
from nevernegative.layers.crop.utils.line import Line, line_intersection
from nevernegative.utils.image import approximate_image_scaling, get_image_corners


class HoughCrop(Cropper):
    def __init__(
        self,
        peak_ratio: float,
        min_distance: int,
        snap_to_edge_map: bool,
        grey_converter: Grey,
        thresholder: ThresholdConfig | Threshold,
        edge_detector: EdgeDetectConfig | EdgeDetect,
        *,
        hough_transform_parameters: HoughTransformParameters = HoughTransformParameters(),
    ) -> None:
        self.peak_ratio = peak_ratio
        self.min_distance = min_distance
        self.snap_to_edge_map = snap_to_edge_map

        if isinstance(grey_converter, LayerConfig):
            grey_converter = grey_converter.initialize()

        if isinstance(edge_detector, LayerConfig):
            edge_detector = edge_detector.initialize()

        if isinstance(thresholder, LayerConfig):
            thresholder = thresholder.initialize()

        self.grey_converter = grey_converter
        self.edge_detector = edge_detector
        self.thresholder = thresholder

        self.hough_transform_parameters = hough_transform_parameters

    def find_bounding_lines(
        self,
        image: NDArray[np.bool],
    ) -> tuple[tuple[Line, Line], tuple[Line, Line]]:
        hspace, angles, distances = ski.transform.hough_line(
            image,
            theta=np.linspace(
                start=self.hough_transform_parameters.start_angle,
                stop=self.hough_transform_parameters.end_angle,
                num=self.hough_transform_parameters.step,
                endpoint=False,
            ),
        )

        peaks = ski.transform.hough_line_peaks(
            hspace,
            angles,
            distances,
            threshold=(self.peak_ratio * np.max(hspace)),
            num_peaks=4,
            min_distance=self.min_distance,
        )

        verticals: list[Line] = []
        horizontals: list[Line] = []

        # Warn if number of peaks is less than 4.
        for _, angle, distance in zip(*peaks):
            line = Line(
                slope=np.tan(angle + np.pi / 2),
                coord=distance * np.array([np.cos(angle), np.sin(angle)]),
                is_vertical=round(2 * angle / np.pi) == 0,
            )

            if line.is_vertical:
                verticals.append(line)
            else:
                horizontals.append(line)

        if len(verticals) != 2 or len(horizontals) != 2:
            raise RuntimeError()

        [left, right] = sorted(verticals, key=lambda line: line.coord[1])
        [top, bottom] = sorted(horizontals, key=lambda line: line.coord[0])

        return (left, right), (top, bottom)

    def find_corners(self, image: NDArray[np.bool]) -> tuple[NDArray[np.float64], tuple[int, int]]:
        verticals, horizontals = self.find_bounding_lines(image)

        # Order is guaranteed to be:
        # [top-left, top-right, bottom-left, bottom-right]
        corners = np.array(
            [
                line_intersection(horizontal, vertical)
                for horizontal, vertical in itertools.product(horizontals, verticals)
            ],
            dtype=np.float64,
        )

        if self.snap_to_edge_map:
            corners = self._snap_corners_to_edge_map(corners, image)

        crop_height = np.linalg.norm(corners[3] - corners[0])
        crop_width = np.linalg.norm(corners[1] - corners[0])

        shape = (int(crop_height), int(crop_width))

        return corners, shape

    @staticmethod
    def _snap_corners_to_edge_map(
        corners: NDArray[np.float64], edge_map: NDArray[np.bool]
    ) -> NDArray[np.float64]:
        edge_pixels = np.flip(np.argwhere(edge_map), axis=1)
        vectors = np.expand_dims(corners, axis=1) - edge_pixels
        distances = np.linalg.vector_norm(vectors, axis=2)

        return edge_pixels[np.argmin(distances, axis=1)].astype(np.float64)

    def __call__(self, image: NDArray) -> NDArray:
        grey = self.grey_converter(image)
        threshold = self.thresholder(grey)
        edge_map = self.edge_detector(threshold)

        crop_corners, crop_shape = self.find_corners(edge_map)
        crop_corners *= approximate_image_scaling(edge_map.shape, image.shape)

        transform = ski.transform.ProjectiveTransform()
        is_success = transform.estimate(
            src=crop_corners,
            dst=get_image_corners(image).astype(np.float64),
        )

        if not is_success:
            raise RuntimeError()

        warped = ski.transform.warp(image, transform.inverse)

        return ski.transform.resize(warped, crop_shape)  # type: ignore
