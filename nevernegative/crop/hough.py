import itertools
from typing import Sequence

import numpy as np
import skimage as ski

from nevernegative.crop.base import Cropper
from nevernegative.crop.config.hough import HoughTransformConfig
from nevernegative.crop.utils.line import Line, line_intersection
from nevernegative.layers.base import Layer
from nevernegative.layers.config.edge import EdgeDetectConfig
from nevernegative.layers.config.threshold import ThresholdConfig
from nevernegative.layers.grey import Grey
from nevernegative.typing.image import EdgeMap, Image, ScalarTypeT


class HoughCrop(Cropper):
    def __init__(
        self,
        peak_ratio: float,
        min_distance: int,
        snap_to_edge_map: bool,
        hough_transform: HoughTransformConfig,
        edge_detector: EdgeDetectConfig,
        thresholder: ThresholdConfig,
        layers: Sequence[Layer] | None = None,
    ) -> None:
        super().__init__(layers)

        self.peak_ratio = peak_ratio
        self.min_distance = min_distance
        self.snap_to_edge_map = snap_to_edge_map
        self.hough_transform = hough_transform

        self.to_grey = Grey()
        self.edge_detector = edge_detector.initialize()
        self.thresholder = thresholder.initialize()

    def _approximate_bounding_lines(
        self,
        image: EdgeMap,
    ) -> tuple[tuple[Line, Line], tuple[Line, Line]]:
        hspace, angles, distances = ski.transform.hough_line(
            image,
            theta=np.linspace(
                start=self.hough_transform.start_angle,
                stop=self.hough_transform.end_angle,
                num=self.hough_transform.step,
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

            if line.is_vertical:  # is horizontal
                verticals.append(line)
            else:
                horizontals.append(line)

        if len(verticals) != 2 or len(horizontals) != 2:
            raise RuntimeError()

        [left, right] = sorted(verticals, key=lambda line: line.coord[1])
        [top, bottom] = sorted(horizontals, key=lambda line: line.coord[0])

        return (left, right), (top, bottom)

    @staticmethod
    def _snap_corner_to_edge_map(corners: np.ndarray, edge_map: EdgeMap) -> np.ndarray:
        edge_pixels = np.flip(np.argwhere(edge_map > 0), axis=1)
        vectors = np.expand_dims(corners, axis=1) - edge_pixels
        distances = np.linalg.vector_norm(vectors, axis=2)

        return edge_pixels[np.argmin(distances, axis=1)].tolist()

    @staticmethod
    def _approximate_image_scale(
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
    ) -> tuple[float, float]:
        input_height, input_width, *_ = input_shape
        output_height, output_width, *_ = output_shape

        return (output_height / input_height, output_width / input_width)

    def _get_crop_corners(self, image: EdgeMap) -> np.ndarray:
        verticals, horizontals = self._approximate_bounding_lines(image)

        corners = np.array(
            [
                line_intersection(horizontal, vertical)
                for horizontal, vertical in itertools.product(horizontals, verticals)
            ]
        )

        if self.snap_to_edge_map:
            corners = self._snap_corner_to_edge_map(corners, image)

        return corners

    def _get_crop_size(self, crop_corners: np.ndarray) -> tuple[int, int]:
        [tl, tr, bl, _] = crop_corners.tolist()

        return (int(np.linalg.norm(bl - tl)), int(np.linalg.norm(tr - tl)))

    def _get_image_corners(self, image: Image) -> np.ndarray:
        return np.array(
            [
                (0, 0),
                (0, image.shape[1]),
                (image.shape[0], 0),
                (image.shape[0], image.shape[1]),
            ]
        )

    def compute(self, image: Image[ScalarTypeT]) -> Image[ScalarTypeT]:
        grey = self.to_grey(image)

        threshold = self.thresholder(grey)
        edge_map = self.edge_detector(threshold)

        crop_corners = self._get_crop_corners(edge_map)
        crop_size = self._get_crop_size(crop_corners)

        crop_corners *= self._approximate_image_scale(edge_map.shape, image.shape)

        image_corners = self._get_image_corners(image)

        transform = ski.transform.ProjectiveTransform()
        is_success = transform.estimate(crop_corners, image_corners)

        if not is_success:
            raise RuntimeError()

        warped = ski.transform.warp(image, transform.inverse)

        return ski.transform.resize(warped, crop_size)  # type: ignore
