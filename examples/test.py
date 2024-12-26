from pathlib import Path

import skimage as ski
from numpy.typing import NDArray

from nevernegative.layers.color.histogram import HistogramBalancer
from nevernegative.layers.color.presets import COLOR_PLUS_200_ADJUSTED
from nevernegative.layers.common.blur import Blur
from nevernegative.layers.common.grey import Grey
from nevernegative.layers.common.resize import Resize
from nevernegative.layers.crop.hough import HoughCrop
from nevernegative.scanner.simple import SimpleScanner


def rotate_180(image: NDArray) -> NDArray:
    return ski.transform.rotate(image, angle=180)


scanner = SimpleScanner(
    # dewarper=HoughDewarper(
    #     num_points=100,
    #     method="linear",
    #     lengthscale="x",
    #     k=2,
    #     preprocessing_layers=[
    #         rotate_180,
    #         Resize(height=800),
    #         Grey(),
    #         Blur(sigma=(3, 3)),
    #     ],
    #     plot_path=Path("results/dewarper"),
    # ),
    cropper=HoughCrop(
        min_distance=30,
        peak_ratio=0.2,
        snap_to_edge_map=False,
        preprocessing_layers=[
            Resize(height=800),
            Grey(),
            Blur(sigma=(2, 2)),
        ],
        plot_path=Path("results/cropper"),
        offset=(5, 5),
    ),
    color_balancer=HistogramBalancer(
        preset=COLOR_PLUS_200_ADJUSTED,
        plot_path=Path("results/balancer"),
    ),
)

scanner.glob(
    source="/Users/louismanestar/Documents/Projects/Film Scanner/nevernegative/images/nikon/*45.NEF",
    destination="/Users/louismanestar/Documents/Projects/Film Scanner/nevernegative/images/nikon/results",
    is_raw=True,
)

# scanner.file(
#     source="/Users/louismanestar/Documents/Projects/Film Scanner/nevernegative/images/nikon/IMG_5031.CR2",
#     destination="/Users/louismanestar/Documents/Projects/Film Scanner/nevernegative/images/nikon/results_2/",
#     is_raw=True,
# )
