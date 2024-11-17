from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field


class ChannelPreset(BaseModel):
    bounds: tuple[float, float] = (0.0, 1.0)
    brightness: float = 0.0
    contrast: float = 0.0


class FilmPreset(BaseModel):
    red: ChannelPreset = Field(default_factory=ChannelPreset)
    green: ChannelPreset = Field(default_factory=ChannelPreset)
    blue: ChannelPreset = Field(default_factory=ChannelPreset)

    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0

    is_negative: bool
    is_monochrome: bool

    def iter_channels(self) -> Iterator[ChannelPreset]:
        for channel in (self.red, self.green, self.blue):
            yield channel

    @property
    def channelwise_brightness(self) -> NDArray[np.float64]:
        return (
            np.array(
                [self.red.brightness, self.green.brightness, self.blue.brightness],
                dtype=np.float64,
            )
            + self.brightness
        )

    @property
    def channelwise_contrast(self) -> NDArray[np.float64]:
        return (
            np.array(
                [self.red.contrast, self.green.contrast, self.blue.contrast],
                dtype=np.float64,
            )
            + self.contrast
        )


COLOR_PLUS_200 = FilmPreset(
    red=ChannelPreset(
        bounds=(0.03, 1.0),
        brightness=0.1,
    ),
    green=ChannelPreset(
        bounds=(0.03, 1),
        brightness=0.05,
    ),
    blue=ChannelPreset(
        bounds=(0.05, 1),
        brightness=-0.1,
        contrast=-0.5,
    ),
    saturation=0.03,
    is_negative=True,
    is_monochrome=False,
)
