from pathlib import Path
from typing import Callable

import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.balancing.histogram_scaling import HistogramScaling
from nevernegative.layers.balancing.presets.film.bw import DELTA_100
from nevernegative.layers.common.grey import Grey
from nevernegative.layers.common.positive import Positive
from nevernegative.layers.common.temperature import AdjustTemperature
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
        # Resize(height=800),
        # HoughCrop(
        #     min_distance=30,
        #     peak_ratio=0.2,
        #     snap_to_edge_map=True,
        #     padding=0.03,
        #     resize=800,
        #     edge_sigma=2,
        #     plot_path=Path("results/cropper"),
        #     # offset=(5, 5),
        # ),
        AdjustTemperature(temperature=5600, plot_path=Path("results/soho/temperature")),
        # PigmentCorrection(film=ULTRAMAX_400, plot_path=Path("results/pigment")),
        Positive(is_negative=True),
        HistogramScaling(
            bounds=(0.01, 0.99),
            invert=DELTA_100.is_negative,
            clip=True,
            plot_path=Path("results/soho/histogram_scaler"),
        ),
        Grey(channel=2),
        # BasicBalance(
        #     film=DELTA_100,
        #     invert=False,
        #     plot_path=Path("results/basic_balance"),
        # ),
        # DenoiseChannels(weight=(0, 0, 0.1), plot_path=Path("results/denoise")),
    ]
)

scanner.glob(
    source="/Users/louismanestar/Documents/Projects/Film Scanner/nevernegative/images/Soho/*.NEF",
    destination="/Users/louismanestar/Documents/Projects/Film Scanner/nevernegative/images/Soho/results",
    is_raw=True,
)
