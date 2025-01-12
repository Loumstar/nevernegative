from pathlib import Path
from typing import Callable

import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.balancing.pigment import PigmentCorrection
from nevernegative.layers.balancing.presets.film.colour import ULTRAMAX_400
from nevernegative.layers.balancing.temperature import AdjustTemperature
from nevernegative.layers.crop.hough import HoughCrop
from nevernegative.layers.post.denoise import DenoiseChannels
from nevernegative.scanner.simple import SimpleScanner


def rotate(angle: float) -> Callable[[NDArray], NDArray]:
    def f(image: NDArray) -> NDArray:
        return ski.transform.rotate(image, angle=angle)

    return f


scanner = SimpleScanner(
    [
        # dewarper=HoughDewarper(
        #     num_points=100,
        #     method="linear",
        #     lengthscale="x",
        #     k=2,
        #     plot_path=Path("results/dewarper"),
        # ),
        # rotate(angle=90),
        HoughCrop(
            min_distance=30,
            peak_ratio=0.2,
            snap_to_edge_map=False,
            padding=0.03,
            resize=800,
            edge_sigma=2,
            plot_path=Path("results/cropper"),
            offset=(5, 5),
        ),
        AdjustTemperature(temperature=5550, plot_path=Path("results/temperature")),
        PigmentCorrection(film=ULTRAMAX_400, plot_path=Path("results/pigment")),
        DenoiseChannels(weight=(0, 0, 0.5), plot_path=Path("results/denoise")),
        # HistogramScaling(
        #     bounds=(0.01, 0.99),
        #     invert=ULTRAMAX_400.is_negative,
        #     clip=False,
        #     plot_path=Path("results/histogram_scaler"),
        # ),
        # BasicBalance(
        #     film=ULTRAMAX_400,
        #     invert=False,
        #     plot_path=Path("results/basic_balance"),
        # ),
    ]
)

scanner.glob(
    source="/Users/louismanestar/Documents/Projects/Film Scanner/nevernegative/images/cleaning/*.NEF",
    destination="/Users/louismanestar/Documents/Projects/Film Scanner/nevernegative/images/cleaning/results_pigment",
    is_raw=True,
)
