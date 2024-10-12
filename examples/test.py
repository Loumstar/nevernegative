from nevernegative.callbacks.save import SaveImageCallback
from nevernegative.layers.color.config.histogram_scaling import HistogramScalingColorBalancerConfig
from nevernegative.layers.common.blur import Blur
from nevernegative.layers.common.config.blur import BlurConfig
from nevernegative.layers.common.config.edge import EdgeDetectConfig
from nevernegative.layers.common.config.resize import ResizeConfig
from nevernegative.layers.common.config.threshold import ThresholdConfig
from nevernegative.layers.common.resize import Resize
from nevernegative.layers.crop.config.hough import HoughCropConfig
from nevernegative.layers.crop.hough import HoughCrop
from nevernegative.layers.dewarp.config.hough import HoughTransformDewarperConfig
from nevernegative.scanner.config.simple import SimpleScannerConfig

config = SimpleScannerConfig(
    layers=[
        Resize(ratio=0.5),
        Blur(sigma=(9, 9)),
        HoughCrop(
            min_distance=30,
            peak_ratio=0.2,
        ),
    ]
)


config = SimpleScannerConfig(
    dewarping=HoughTransformDewarperConfig(num_points=10, batch_average=False, center="center"),
    cropping=HoughCropConfig(
        min_distance=30,
        hough_peak_ratio=0.2,
        thresholder=ThresholdConfig(type="mean"),
        edge_detector=EdgeDetectConfig(type="canny"),
        transformations=[
            ResizeConfig(ratio=0.5),
            BlurConfig((9, 9)),
        ],
    ),
    color_balancing=HistogramScalingColorBalancerConfig(
        invert=True,
        padding=0.1,
    ),
    pool_size=10,
    cache_intermediate=False,
    callbacks=[SaveImageCallback("results/", suffix=".png")],
)
