from typing import Literal

from nevernegative.layers.color.config.base import ColorBalancerConfig
from nevernegative.layers.color.histogram_scaling import HistogramScalingColorBalancer


class HistogramScalingColorBalancerConfig(ColorBalancerConfig[HistogramScalingColorBalancer]):
    type: Literal["histogram_scaling"] = "histogram_scaling"
