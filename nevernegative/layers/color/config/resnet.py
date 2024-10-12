from typing import Literal

from nevernegative.layers.color.config.base import ColorBalancerConfig
from nevernegative.layers.color.resnet import ResNetColorBalancer


class ResNetColorBalancerConfig(ColorBalancerConfig[ResNetColorBalancer]):
    type: Literal["resnet"] = "resnet"
