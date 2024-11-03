from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from nevernegative.layers.base import Layer
from nevernegative.layers.common.edge import EdgeDetect
from nevernegative.layers.common.grey import Grey
from nevernegative.layers.common.threshold import Threshold
from nevernegative.layers.dewarp.base import Dewarper
from nevernegative.layers.typing import LayerCallableT
from nevernegative.layers.utils.corner_detection.hough import HoughTransform
from nevernegative.layers.utils.decorators import save_figure


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
        self.method = method
        self.k = k

        self.lengthscale = lengthscale

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

        [cx, cy] = self._get_image_center(image, format="cartesian")

        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

        if self.lengthscale == "x":
            euclidean = (((x - cx) / cx) ** 2) + (((y - cy) / cx) ** 2)
        elif self.lengthscale == "y":
            euclidean = (((x - cx) / cy) ** 2) + (((y - cy) / cy) ** 2)
        if self.lengthscale == "xy":
            euclidean = (((x - cx) / cx) ** 2) + (((y - cy) / cy) ** 2)

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

    def sample_coordinates(self, c1: NDArray, c2: NDArray) -> NDArray:
        x1, y1 = c1
        x2, y2 = c2

        x_sample = np.linspace(x1, x2, num=self.num_points, endpoint=True)
        y_sample = np.linspace(y1, y2, num=self.num_points, endpoint=True)

        return np.stack([x_sample, y_sample], axis=-1)

    def _intersect(
        self,
        lines: NDArray,
        *,
        bounds: NDArray | None = None,
        eps: float = 1e-9,
    ) -> NDArray:
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

        if bounds is not None:
            x_bounds, y_bounds = np.moveaxis(bounds, 2, 0)
            x_bounds = np.sort(x_bounds, axis=1)
            y_bounds = np.sort(y_bounds, axis=1)

            # If it falls outside of the bounds, set the result to NaN
            x[np.logical_or(x < x_bounds[:, 0], x > x_bounds[:, 1])] = np.nan
            y[np.logical_or(y < y_bounds[:, 0], y > y_bounds[:, 1])] = np.nan

        return np.stack((x, y), axis=-1)

    def _sample_undistorted_points(self, corners: NDArray, image: NDArray) -> NDArray:
        vectors = corners[:, np.newaxis] - corners  # NxNx2
        distances = np.linalg.norm(vectors, axis=2)  # NxX
        distances[distances == 0] = np.inf

        neighbour_1, neighbour_2 = np.argsort(distances, axis=1).T[:2]  # each N
        corner_indices = np.arange(corners.shape[0])  # N

        neighbour_1_indices = np.stack((corner_indices, neighbour_1), axis=-1)  # Nx2
        neighbour_2_indices = np.stack((corner_indices, neighbour_2), axis=-1)  # Nx2

        pair_indices = np.concatenate((neighbour_1_indices, neighbour_2_indices), axis=0)  # 2Nx2
        pair_indices = np.unique(np.sort(pair_indices, axis=1), axis=0)  # Mx2

        source, destination = corners[pair_indices].transpose((1, 0, 2))  # each Mx2

        if self.method == "linear":
            t = np.linspace(0, 1, num=self.num_points // 4, endpoint=False)
            samples = source[..., np.newaxis] + (destination - source)[..., np.newaxis] * t

            return np.swapaxes(samples, 1, 2).reshape(-1, 2)

        vector = destination - source  # Mx2

        slope = vector[:, 1] / vector[:, 0]  # M

        corner_lines = np.concatenate((source, slope[:, np.newaxis]), axis=1)  # Mx3

        center = self._get_image_center(image, format="cartesian")

        angles = np.tan(np.linspace(0, np.pi, num=self.num_points // 2, endpoint=False))
        radial_source = np.tile(center, reps=(self.num_points // 2, 1))
        radial_lines = np.concatenate((radial_source, angles[:, np.newaxis]), axis=1)  # Mx3

        corner_line_indices = np.arange(corner_lines.shape[0])
        radial_line_indices = np.arange(radial_lines.shape[0])

        # Create combinations of line indices and filter combinations where it is the same line.
        combinations = np.array(np.meshgrid(corner_line_indices, radial_line_indices))
        combinations = combinations.T.reshape(-1, 2)

        line_pairs = np.stack(
            (corner_lines[combinations[:, 0]], radial_lines[combinations[:, 1]]), axis=1
        )

        bounds = corners[pair_indices[combinations[:, 0]]]

        points = self._intersect(line_pairs, bounds=bounds)
        self.plot("radial.png", image, lines=radial_lines)
        mask = np.logical_or(np.isnan(points[:, 0]), np.isnan(points[:, 1]))
        points = points[~mask]

        return points

    def intersect_with_edge_map(
        self, undistorted_points: NDArray, image: NDArray[np.bool], hough_transform: HoughTransform
    ) -> NDArray:
        if self.method == "linear":
            return hough_transform.snap(undistorted_points)

        edge_pixels = np.flip(np.argwhere(image), axis=1)  # M,2
        center = self._get_image_center(image, format="cartesian")

        vectors = center - undistorted_points  # Nx2
        euclidean = np.linalg.norm(vectors, axis=1)  # N
        distortion_vector = edge_pixels[:, np.newaxis] - undistorted_points  # MxNx2

        t = np.sum(distortion_vector * vectors, axis=2) / (euclidean**2)  # MxN

        projections: NDArray = (t[..., np.newaxis] * vectors) + undistorted_points[: np.newaxis]
        score = np.linalg.norm(projections - edge_pixels[:, np.newaxis], axis=2)
        score += 10 * np.abs(t)  # MxN

        return edge_pixels[np.argmin(score, axis=0)]

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

    def _estimate_k(
        self,
        undistorted_points: NDArray,
        distorted_points: NDArray,
        image: NDArray,
    ) -> NDArray:
        center = self._get_image_center(image, format="cartesian")

        if self.lengthscale == "x":
            lengthscale = center[0]
        elif self.lengthscale == "y":
            lengthscale = center[1]
        if self.lengthscale == "xy":
            lengthscale = center

        vector = (undistorted_points - distorted_points) / lengthscale
        radii = np.linalg.norm((distorted_points - center) / lengthscale, axis=1)

        difference = np.linalg.norm(vector, axis=1)
        powers = np.arange(self.k) + 1
        radius_matrix = radii[..., np.newaxis] ** powers
        # radius_matrix = np.stack([radii**1, radii**2, radii**3])

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
        if self.lengthscale == "x":
            lengthscale = center[0]
        elif self.lengthscale == "y":
            lengthscale = center[1]
        if self.lengthscale == "xy":
            lengthscale = center

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

        corners = hough_transform.corners
        self.plot(
            "hough",
            preprocessed_image,
            lines=hough_transform.lines,
            undistorted_points=corners,
        )

        undistorted_points = self._sample_undistorted_points(corners, preprocessed_image)
        distorted_points = self.intersect_with_edge_map(
            undistorted_points, preprocessed_image, hough_transform
        )

        k = self._estimate_k(undistorted_points, distorted_points, preprocessed_image)

        self.plot(
            "points",
            preprocessed_image,
            distorted_points=distorted_points,
            undistorted_points=undistorted_points,
        )

        self.plot_warping_contour("contour.png", image, k)
        self.plot_warping_multiplier("multiplier.png", k)

        # FIXME: handle which way to warp based on whether distortion is inside the box or not.
        distorted = ski.transform.warp(
            image,
            inverse_map=self._inverse_map,
            order=1,
            map_args={
                "k": k,
                "invert": False,
                "center": self._get_image_center(image, format="cartesian"),
            },
        )
        # distorted = self.barrel_warp(image, k1, k2, k3, unwarp=False)

        self.plot("distorted", image)
        self.plot("undistorted", distorted)
        self.plot_image_overlay("overlay", image, distorted)

        return distorted
