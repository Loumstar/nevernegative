from typing import Literal

from pydantic import Field

from nevernegative.layers.config.base import LayerConfig
from nevernegative.layers.threshold import Threshold


class ThresholdConfig(LayerConfig[Threshold]):
    type: Literal["threshold"] = Field("threshold")

    def initialize(self) -> Threshold:
        return Threshold()
