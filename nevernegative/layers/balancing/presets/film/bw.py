from typing import Literal

from nevernegative.layers.balancing.presets.film.base import Film


class BlackAndWhiteNegative(Film):
    is_negative: Literal[True] = True
    is_colour: Literal[False] = False


DELTA_100 = BlackAndWhiteNegative(
    brightness=-0.05,
    contrast=-0.4,
    grey_channel=2,
)
