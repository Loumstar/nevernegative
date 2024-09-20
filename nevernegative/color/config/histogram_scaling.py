from typing import Literal

from nevernegative.color.config.base import ColorBalancerConfig
from nevernegative.color.histogram_scaling import HistogramScalingColorBalancer


class HistogramScalingColorBalancerConfig(ColorBalancerConfig[HistogramScalingColorBalancer]):
    type: Literal["histogram_scaling"] = "histogram_scaling"

    def initialize(self) -> HistogramScalingColorBalancer:
        return HistogramScalingColorBalancer(
            preprocessing_layers=self.initialize_preprocessing_layers(),  # type: ignore[abstract]
        )
