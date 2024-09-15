from typing import Annotated, TypeAlias

from pydantic import Field

from nevernegative.color.config.base import ColorBalancerConfig
from nevernegative.crop.config.base import CropperConfig
from nevernegative.dewarp.config.base import DewarperConfig
from nevernegative.layers.config.blur import BlurConfig
from nevernegative.layers.config.edge import EdgeDetectConfig
from nevernegative.layers.config.grey import GreyConfig
from nevernegative.layers.config.resize import ResizeConfig
from nevernegative.layers.config.threshold import ThresholdConfig

LayerConfigs: TypeAlias = Annotated[
    BlurConfig | EdgeDetectConfig | GreyConfig | ResizeConfig | ThresholdConfig,
    Field(discriminator="type"),
]

DewarperConfigs: TypeAlias = Annotated[DewarperConfig, Field(discriminator="type")]
CropperConfigs: TypeAlias = Annotated[CropperConfig, Field(discriminator="type")]
ColorBalancerConfigs: TypeAlias = Annotated[ColorBalancerConfig, Field(discriminator="type")]
