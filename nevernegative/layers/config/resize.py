from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from nevernegative.layers.config.base import LayerConfig
from nevernegative.layers.resize import Resize


class ResizeConfig(LayerConfig[Resize]):
    type: Literal["resize"] = Field("resize")

    height: int | None = None
    width: int | None = None

    ratio: float | None = None

    anti_aliasing: bool = True

    @model_validator(mode="after")
    def check_resizing_is_set(self) -> Self:
        if all(value is None for value in (self.height, self.width, self.ratio)):
            raise ValueError("At least one of height, width or ratio must be set.")

        if self.ratio is not None:
            if self.height is not None:
                raise ValueError("Cannot set ratio and height.")

            if self.width is not None:
                raise ValueError("Cannot set ratio and width.")

        return self

    def initialize(self) -> Resize:
        return Resize(
            height=self.height,
            width=self.width,
            ratio=self.ratio,
            anti_aliasing=self.anti_aliasing,
        )
