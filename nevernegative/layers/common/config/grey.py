from typing import Literal

from pydantic import Field

from nevernegative.layers.common.grey import Grey
from nevernegative.layers.config import LayerConfig


class GreyConfig(LayerConfig):
    type: Literal["grey"] = Field(default="grey")

    def initialize(self) -> Grey:
        return Grey()
