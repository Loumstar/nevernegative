from typing import Annotated, TypeAlias

from pydantic import Field

from nevernegative.color.config.histogram_scaling import HistogramScalingColorBalancerConfig
from nevernegative.color.config.resnet import ResNetColorBalancerConfig
from nevernegative.crop.config.hough import HoughCropConfig
from nevernegative.dewarp.config.checkerboard import CheckerboardDewarperConfig
from nevernegative.dewarp.config.hough import HoughTransformDewarperConfig
from nevernegative.layers.config.blur import BlurConfig
from nevernegative.layers.config.edge import EdgeDetectConfig
from nevernegative.layers.config.grey import GreyConfig
from nevernegative.layers.config.resize import ResizeConfig
from nevernegative.layers.config.threshold import ThresholdConfig

LayerConfigs: TypeAlias = Annotated[
    BlurConfig | EdgeDetectConfig | GreyConfig | ResizeConfig | ThresholdConfig,
    Field(discriminator="type"),
]

DewarperConfigs: TypeAlias = Annotated[
    HoughTransformDewarperConfig | CheckerboardDewarperConfig,
    Field(discriminator="type"),
]

CropperConfigs: TypeAlias = Annotated[
    HoughCropConfig,
    Field(discriminator="type"),
]

ColorBalancerConfigs: TypeAlias = Annotated[
    HistogramScalingColorBalancerConfig | ResNetColorBalancerConfig,
    Field(discriminator="type"),
]
