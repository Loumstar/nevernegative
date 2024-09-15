from typing import Literal

from pydantic import Field

from nevernegative.layers.config.base import LayerConfig
from nevernegative.layers.grey import Grey


class GreyConfig(LayerConfig):
    type: Literal["grey"] = Field("grey")

    def initialize(self) -> Grey:
        return Grey()
