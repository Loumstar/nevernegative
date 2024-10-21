from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from nevernegative.layers.base import Layer
from nevernegative.layers.common.edge import EdgeDetect
from nevernegative.layers.common.grey import Grey
from nevernegative.layers.common.threshold import Threshold
from nevernegative.layers.crop.base import Cropper
from nevernegative.layers.utils.corner_detection.hough import HoughTransform
from nevernegative.layers.utils.line import Line


class HoughCrop(Cropper):
    def __init__(
        self,
        *,
        peak_ratio: float = 0.2,
        min_distance: int = 30,
        snap_to_edge_map: bool = True,
        preprocessing_layers: Sequence[Layer] | None = None,
        edge_sigma: float = 1.0,
        edge_low_threshold: float | None = None,
        edge_high_threshold: float | None = None,
        start_angle: float = np.deg2rad(-45),
        end_angle: float = np.deg2rad(135),
        step: int = 360,
    ) -> None:
        self.peak_ratio = peak_ratio
        self.min_distance = min_distance
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.step = step

        self.snap_to_edge_map = snap_to_edge_map

        self.preprocessing_layers = list(preprocessing_layers or [])
        self.preprocessing_layers.extend(
            [
                Grey(),
                Threshold(),
                EdgeDetect(
                    sigma=edge_sigma,
                    low_threshold=edge_low_threshold,
                    high_threshold=edge_high_threshold,
                ),
            ]
        )

    def save_image(
        self,
        name: str,
        image: NDArray,
        lines: Sequence[Line] = (),
        points: NDArray | None = None,
        *,
        format: Literal["image", "xy"] = "xy",
    ) -> None:
        figure, axis = plt.subplots()

        if image.max() > 1:
            image /= 255

        axis.imshow(image)

        for line in lines:
            if format == "image":
                raise NotImplementedError()

            axis.axline(line.coord, slope=line.slope, color="red")

        if points is not None:
            axis.scatter(*points.T, color="green")

        axis.axis("off")

        Path("results/").mkdir(parents=True, exist_ok=True)
        figure.savefig(f"results/{name}.png")

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

    def __call__(self, image: NDArray) -> NDArray:
        preprocessed_image = image

        for i, layer in enumerate(self.preprocessing_layers):
            preprocessed_image = layer(preprocessed_image)
            self.save_image(f"layer_{i}", preprocessed_image)

        hough_transform = HoughTransform(
            preprocessed_image,
            peak_ratio=self.peak_ratio,
            min_distance=self.min_distance,
            start_angle=self.start_angle,
            end_angle=self.end_angle,
            step=self.step,
            max_num_peaks=4,
        )

        crop_corners = hough_transform.corners(snap_to_edge_map=self.snap_to_edge_map)
        self.save_image(
            "hough", preprocessed_image, lines=hough_transform.lines(), points=crop_corners
        )

        crop_corners = self._sort_coordinates(crop_corners)
        crop_corners *= self._image_scale(preprocessed_image, image)

        image_corners = self._image_corners(image)
        image_corners = self._sort_coordinates(image_corners)

        perspective_transform = ski.transform.ProjectiveTransform()

        is_success = perspective_transform.estimate(
            src=crop_corners.astype(np.float64),
            dst=image_corners.astype(np.float64),
        )

        if not is_success:
            raise RuntimeError()

        warped = ski.transform.warp(image, perspective_transform.inverse)
        self.save_image("transformed", warped)

        shape = self._crop_shape(crop_corners)

        return ski.transform.resize(warped, shape)
