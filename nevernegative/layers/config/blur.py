from typing import Literal

from pydantic import Field

from nevernegative.layers.blur import Blur
from nevernegative.layers.config.base import LayerConfig


class BlurConfig(LayerConfig[Blur]):
    type: Literal["blur"] = Field("blur")

    sigma: int | tuple[int, int]
    mode: Literal["reflect", "nearest", "mirror", "wrap"] = "nearest"

    def initialize(self) -> Blur:
        return Blur(sigma=self.sigma, mode=self.mode)
