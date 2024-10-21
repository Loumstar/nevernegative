from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from nevernegative.layers.base import Layer
from nevernegative.layers.common.edge import EdgeDetect
from nevernegative.layers.common.grey import Grey
from nevernegative.layers.common.threshold import Threshold
from nevernegative.layers.dewarp.base import Dewarper
from nevernegative.layers.utils.corner_detection.hough import HoughTransform
from nevernegative.layers.utils.line import Line


class HoughDewarper(Dewarper):
    def __init__(
        self,
        num_points: int = 2,
        center: tuple[float, float] | Literal["center"] = "center",
        *,
        edge_sigma: float = 1.0,
        edge_low_threshold: float | None = None,
        edge_high_threshold: float | None = None,
        peak_ratio: float = 0.3,
        min_distance: int = 30,
        start_angle: float = np.deg2rad(-45),
        end_angle: float = np.deg2rad(135),
        step: int = 360,
        preprocessing_layers: Sequence[Layer] | None = None,
    ) -> None:
        self.num_points = num_points
        self.center = center

        self.peak_ratio = peak_ratio
        self.min_distance = min_distance
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.step = step

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
            image = image.astype(np.float64) / 255

        axis.imshow(image)

        for line in lines:
            if format == "image":
                raise NotImplementedError()

            axis.axline(line.coord, slope=line.slope, color="red")

        if points is not None:
            if points.shape[1] == 4:
                for [x1, y1, x2, y2] in points:
                    axis.plot([x1, x2], [y1, y2], color="red")

            else:
                axis.scatter(*points.T, color="green")
            # axis.scatter(*points.T, color="green")

        axis.axis("off")

        Path("results/dewarper").mkdir(parents=True, exist_ok=True)
        figure.savefig(f"results/dewarper/{name}.png")

    def sample_coordinates(self, c1: NDArray, c2: NDArray) -> NDArray:
        x1, y1 = c1
        x2, y2 = c2

        x_sample = np.linspace(x1, x2, num=self.num_points, endpoint=True)
        y_sample = np.linspace(y1, y2, num=self.num_points, endpoint=True)

        return np.stack([x_sample, y_sample], axis=-1)

    def barrel_warp(self, image: NDArray, k1, k2, k3, unwarp=True) -> NDArray:
        def inverse_map(xy: NDArray, **_: Any) -> NDArray:
            center = xy.mean(axis=0)
            normalised = (xy - center) / center

            euclidean = np.linalg.norm(normalised, axis=1)
            multiplier = (k1 * euclidean) + (k2 * (euclidean**2)) + (k3 * (euclidean**3))
            print(multiplier.min(), multiplier.max())
            return ((normalised / (1 + multiplier)[:, None]) * center) + center

        return ski.transform.warp(image, inverse_map=inverse_map, order=1)

    def __call__(self, image: NDArray) -> NDArray:
        preprocessed_image = image

        for i, layer in enumerate(self.preprocessing_layers):
            preprocessed_image = layer(preprocessed_image)
            # self.save_image(f"layer_{i}", preprocessed_image)

        hough_transform = HoughTransform(
            preprocessed_image,
            peak_ratio=self.peak_ratio,
            min_distance=self.min_distance,
            start_angle=self.start_angle,
            end_angle=self.end_angle,
            step=self.step,
            max_num_peaks=4,
        )

        corners = hough_transform.corners(snap_to_edge_map=False)

        self.save_image("hough", preprocessed_image, lines=hough_transform.lines(), points=corners)

        vectors = corners[:, np.newaxis] - corners
        distances = np.linalg.norm(vectors, axis=2)

        neighbours = corners[np.argsort(distances, axis=1)]
        samples_list: list[NDArray] = []

        for [corner, n1, n2, *_] in neighbours:
            samples_list.extend(
                [
                    self.sample_coordinates(corner, n1),
                    self.sample_coordinates(corner, n2),
                ]
            )

        samples = np.concat(samples_list, axis=0)
        distorted_samples = hough_transform.snap_to_edge_map(samples)
        self.save_image(
            "points",
            preprocessed_image,
            points=np.concat([samples, distorted_samples], axis=1),
        )

        height, width, *_ = preprocessed_image.shape
        center = np.array([width, height]) / 2

        distorted_vector = (distorted_samples - center) / center
        undistorted_vector = (samples - distorted_samples) / center

        distorted_radii = np.repeat(np.linalg.norm(distorted_vector, axis=1), repeats=2)

        displacement_matrix = (undistorted_vector / distorted_vector).ravel()
        radius_matrix = np.stack([distorted_radii**1, distorted_radii**2, distorted_radii**3])

        [k1, k2, k3], *_ = np.linalg.lstsq(radius_matrix.T, displacement_matrix)

        distorted = self.barrel_warp(image, k1, k2, k3)

        self.save_image("distorted", image)
        self.save_image("undistorted", distorted)

        return distorted
