from typing import Literal

from nevernegative.color.config.base import ColorBalancerConfig
from nevernegative.color.preset import PresetColorBalancer


class PresetColorBalancerConfig(ColorBalancerConfig[PresetColorBalancer]):
    type: Literal["preset"] = "preset"

    def initialize(self) -> PresetColorBalancer:
        return PresetColorBalancer(
            preprocessing_layers=self.initialize_preprocessing_layers(),  # type: ignore[abstract]
        )
