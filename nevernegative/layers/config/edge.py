from typing import Literal

from pydantic import Field

from nevernegative.layers.config.base import LayerConfig
from nevernegative.layers.edge import EdgeDetect


class EdgeDetectConfig(LayerConfig[EdgeDetect]):
    type: Literal["edge_detection"] = Field("edge_detection")

    sigma: float
    low_threshold: float
    high_threshold: float

    def initialize(self) -> EdgeDetect:
        return EdgeDetect(
            sigma=self.sigma,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )
