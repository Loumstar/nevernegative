from nevernegative.layers.color.histogram_scaling import HistogramScalingColorBalancer
from nevernegative.layers.common.blur import Blur
from nevernegative.layers.common.grey import Grey
from nevernegative.layers.common.resize import Resize
from nevernegative.layers.crop.hough import HoughCrop
from nevernegative.layers.dewarp.hough import HoughDewarper
from nevernegative.scanner.simple import SimpleScanner

scanner = SimpleScanner(
    dewarper=HoughDewarper(
        num_points=51,
        center="center",
        preprocessing_layers=[
            Resize(height=800),
            Grey(),
            Blur(sigma=(3, 3)),
        ],
    ),
    cropper=HoughCrop(
        min_distance=30,
        peak_ratio=0.2,
        preprocessing_layers=[
            Resize(height=800),
            Grey(),
            Blur(sigma=(2, 2)),
        ],
    ),
    color_balancer=HistogramScalingColorBalancer(
        red=(0.03, 1),
        green=(0.03, 1),
        blue=(0.05, 1),
        brightness=(0.1, 0.05, -0.1),
        contrast=(0.0, 0.0, -0.5),
    ),
)


scanner.file(
    source="./images/IMG_4863.CR2",
    destination="./images/results/IMG_4863.png",
    return_array=False,
    is_raw=True,
)
