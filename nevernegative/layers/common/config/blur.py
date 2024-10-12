from typing import Literal

from pydantic import Field

from nevernegative.layers.common.blur import Blur
from nevernegative.layers.config import LayerConfig


class BlurConfig(LayerConfig[Blur]):
    type: Literal["blur"] = Field(default="blur")

    sigma: int | tuple[int, int]
    mode: Literal["reflect", "nearest", "mirror", "wrap"] = "nearest"

    def initialize(self) -> Blur:
        return Blur(sigma=self.sigma, mode=self.mode)
