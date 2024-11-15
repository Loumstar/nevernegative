from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from nevernegative.layers import utils
from nevernegative.layers.base import Layer
from nevernegative.layers.common.edge import EdgeDetect
from nevernegative.layers.common.grey import Grey
from nevernegative.layers.common.threshold import Threshold
from nevernegative.layers.dewarp.base import Dewarper
from nevernegative.layers.typing import LayerCallableT
from nevernegative.layers.utils.decorators import save_figure
from nevernegative.layers.utils.hough import HoughTransform


class HoughDewarper(Dewarper):
    def __init__(
        self,
        num_points: int = 2,
        method: Literal["radial", "linear"] = "radial",
        *,
        k: int = 3,
        edge_sigma: float = 1.0,
        edge_low_threshold: float | None = None,
        edge_high_threshold: float | None = None,
        peak_ratio: float = 0.3,
        min_distance: int = 30,
        start_angle: float = np.deg2rad(-45),
        end_angle: float = np.deg2rad(135),
        step: int = 360,
        lengthscale: Literal["x", "y", "xy"] = "x",
        preprocessing_layers: Sequence[Layer | LayerCallableT] | None = None,
        plot_path: Path | None = None,
        figure_size: tuple[int, int] = (15, 15),
    ) -> None:
        super().__init__(plot_path, figure_size)

        self.num_points = num_points
        self.method: Literal["radial", "linear"] = method
        self.k = k

        self.lengthscale: Literal["x", "y", "xy"] = lengthscale

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

    @save_figure
    def plot(
        self,
        image: NDArray,
        *,
        lines: NDArray | None = None,
        undistorted_points: NDArray | None = None,
        distorted_points: NDArray | None = None,
    ) -> Figure:
        figure, axis = plt.subplots()

        if image.max() > 1:
            image = image.astype(np.float64) / 255

        axis.imshow(image)

        if lines is not None:
            for [x, y, slope] in lines:
                axis.axline((x, y), slope=slope, color="red")

        if undistorted_points is not None and distorted_points is not None:
            vectors = undistorted_points - distorted_points
            axis.quiver(*distorted_points.T, *vectors.T, angles="xy")

        if undistorted_points is not None:
            axis.scatter(*undistorted_points.T, color="green")

        if distorted_points is not None:
            axis.scatter(*distorted_points.T, color="blue")

        axis.axis("off")

        return figure

    @save_figure
    def plot_warping_contour(self, image: NDArray, k: NDArray) -> Figure:
        figure, axis = plt.subplots()

        [cx, cy] = center = utils.image.get_center(image, format="cartesian")
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

        lengthscale = utils.image.get_lengthscale(self.lengthscale, image_center=center)
        euclidean = (((x - cx) / lengthscale) ** 2) + (((y - cy) / lengthscale) ** 2)

        powers = np.arange(k.shape[0]) + 1  # k
        euclidean_matrix = euclidean[..., np.newaxis] ** powers  # N x k
        multiplier = np.sum(euclidean_matrix * k, axis=-1)

        axis.imshow(image, alpha=0.5)

        contour = axis.contourf(x, y, multiplier, cmap="Spectral_r", levels=200, alpha=0.5)
        figure.colorbar(contour)

        return figure

    @save_figure
    def plot_warping_multiplier(self, k: NDArray) -> Figure:
        figure, axis = plt.subplots()

        euclidean = np.linspace(0, 1, num=200) ** 2
        powers = np.arange(k.shape[0]) + 1  # k
        euclidean_matrix = euclidean[..., np.newaxis] ** powers  # N x k

        multiplier = np.sum(euclidean_matrix * k, axis=-1)

        axis.plot(euclidean, multiplier)

        return figure

    @save_figure
    def plot_image_overlay(self, distorted: NDArray, undistorted: NDArray) -> Figure:
        figure, axis = plt.subplots()

        distorted_grey = ski.color.rgb2gray(distorted)
        undistorted_grey = ski.color.rgb2gray(undistorted)

        result = np.stack((distorted_grey, undistorted_grey, undistorted_grey), axis=-1)

        axis.imshow(result)
        axis.axis("off")

        return figure

    def _estimate_k(
        self,
        undistorted_points: NDArray,
        distorted_points: NDArray,
        image_center: NDArray,
    ) -> NDArray:
        lengthscale = utils.image.get_lengthscale(self.lengthscale, image_center=image_center)

        vector = (undistorted_points - distorted_points) / lengthscale
        radii = np.linalg.norm((distorted_points - image_center) / lengthscale, axis=1)

        difference = np.linalg.norm(vector, axis=1)
        powers = np.arange(self.k) + 1
        radius_matrix = radii[..., np.newaxis] ** powers

        k, *_ = np.linalg.lstsq(radius_matrix, difference / radii)

        return k

    def _inverse_map(
        self,
        xy: NDArray,
        *,
        k: NDArray,
        center: NDArray,
        invert: bool = False,
    ) -> NDArray:
        lengthscale = utils.image.get_lengthscale(self.lengthscale, image_center=center)

        normalised = (xy - center) / lengthscale
        euclidean = np.linalg.norm(normalised, axis=1)  # N

        powers = np.arange(k.shape[0]) + 1  # k
        euclidean_matrix = euclidean[:, np.newaxis] ** powers  # N x k
        multiplier = np.sum(euclidean_matrix * k, axis=1)

        if invert:
            multiplier *= -1

        return ((normalised / (1 - multiplier)[:, None]) * lengthscale) + center

    def __call__(self, image: NDArray) -> NDArray:
        preprocessed_image = image

        for i, layer in enumerate(self.preprocessing_layers):
            preprocessed_image = layer(preprocessed_image)
            self.plot(f"layer_{i}", preprocessed_image)

        hough_transform = HoughTransform(
            preprocessed_image,
            peak_ratio=self.peak_ratio,
            min_distance=self.min_distance,
            start_angle=self.start_angle,
            end_angle=self.end_angle,
            step=self.step,
            max_num_peaks=4,
            snap_corners_to_edge_map=True,
        )

        self.plot(
            "hough",
            preprocessed_image,
            lines=hough_transform.lines,
            undistorted_points=hough_transform.corners,
        )

        center = utils.image.get_center(preprocessed_image, format="cartesian")

        undistorted = utils.corners.sample_bounding_box(
            utils.corners.corner_pairs(hough_transform.corners),
            num_points=self.num_points,
            image_center=center,
            mode=self.method,
        )

        distorted = utils.snap.snap_to_edge_map(
            undistorted,
            preprocessed_image,
            method=self.method,
        )

        self.plot(
            "points",
            preprocessed_image,
            distorted_points=distorted,
            undistorted_points=undistorted,
        )

        k = self._estimate_k(undistorted, distorted, center)

        self.plot_warping_contour("contour.png", image, k)
        self.plot_warping_multiplier("multiplier.png", k)

        # FIXME: handle which way to warp based on whether distortion is inside the box or not.
        fixed = ski.transform.warp(
            image,
            inverse_map=self._inverse_map,
            order=1,
            map_args={
                "k": k,
                "invert": False,
                "center": utils.image.get_center(image, format="cartesian"),
            },
        )

        self.plot("distorted", image)
        self.plot("undistorted", fixed)
        self.plot_image_overlay("overlay", image, fixed)

        return fixed
