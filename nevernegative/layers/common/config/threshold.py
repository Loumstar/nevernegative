from typing import Literal

from pydantic import Field

from nevernegative.layers.common.threshold import Threshold
from nevernegative.layers.config import LayerConfig


class ThresholdConfig(LayerConfig[Threshold]):
    type: Literal["threshold"] = Field(default="threshold")

    def initialize(self) -> Threshold:
        return Threshold()
