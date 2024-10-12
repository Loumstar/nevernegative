from typing import Literal

from pydantic import Field

from nevernegative.layers.common.edge import EdgeDetect
from nevernegative.layers.config import LayerConfig


class EdgeDetectConfig(LayerConfig[EdgeDetect]):
    type: Literal["edge_detection"] = Field(default="edge_detection")

    sigma: float
    low_threshold: float
    high_threshold: float

    def initialize(self) -> EdgeDetect:
        return EdgeDetect(
            sigma=self.sigma,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )
