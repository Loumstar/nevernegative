from pathlib import Path

import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.balancing.basic import BasicBalance
from nevernegative.layers.balancing.histogram_scaling import HistogramScaling
from nevernegative.layers.balancing.presets.film.bw import DELTA_100
from nevernegative.layers.crop.hough import HoughCrop
from nevernegative.scanner.simple import SimpleScanner


def rotate_180(image: NDArray) -> NDArray:
    return ski.transform.rotate(image, angle=180)


scanner = SimpleScanner(
    [
        # dewarper=HoughDewarper(
        #     num_points=100,
        #     method="linear",
        #     lengthscale="x",
        #     k=2,
        #     plot_path=Path("results/dewarper"),
        # ),
        rotate_180,
        HoughCrop(
            min_distance=30,
            peak_ratio=0.2,
            snap_to_edge_map=False,
            resize=800,
            edge_sigma=2,
            plot_path=Path("results/cropper"),
            offset=(5, 5),
        ),
        HistogramScaling(
            bounds=(0.01, 0.99),
            invert=DELTA_100.is_negative,
            clip=False,
            plot_path=Path("results/histogram_scaler"),
        ),
        BasicBalance(
            film=DELTA_100,
            invert=False,
            plot_path=Path("results/basic_balance"),
        ),
    ]
)

scanner.glob(
    source="/Users/louismanestar/Documents/Projects/Film Scanner/nevernegative/images/delta_3200_v2/*.NEF",
    destination="/Users/louismanestar/Documents/Projects/Film Scanner/nevernegative/images/delta_3200_v2/results",
    is_raw=True,
)
