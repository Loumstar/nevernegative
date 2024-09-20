from typing import Literal

from nevernegative.color.config.base import ColorBalancerConfig
from nevernegative.color.resnet import ResNetColorBalancer


class ResNetColorBalancerConfig(ColorBalancerConfig[ResNetColorBalancer]):
    type: Literal["resnet"] = "resnet"

    def initialize(self) -> ResNetColorBalancer:
        return ResNetColorBalancer(
            preprocessing_layers=self.initialize_preprocessing_layers(),  # type: ignore[abstract]
        )
