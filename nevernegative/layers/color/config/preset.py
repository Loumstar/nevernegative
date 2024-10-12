from typing import Literal

from nevernegative.layers.color.config.base import ColorBalancerConfig
from nevernegative.layers.color.preset import PresetColorBalancer


class PresetColorBalancerConfig(ColorBalancerConfig[PresetColorBalancer]):
    type: Literal["preset"] = "preset"
