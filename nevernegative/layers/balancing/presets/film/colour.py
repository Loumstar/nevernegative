from typing import Literal

from nevernegative.layers.balancing.presets.film.base import Film


class ColourNegative(Film):
    is_negative: Literal[True] = True
    is_colour: Literal[True] = True


COLOR_PLUS_200 = ColourNegative(brightness=(0.1, 0.05, 0.1), saturation=0.03)
ULTRAMAX_400 = ColourNegative(brightness=(0, -0.03, -0.1), contrast=(0, 0, 0.4))

HANALOGITAL_FLOURITE = ColourNegative(brightness=(-0.05, 0.05, -0.1), contrast=(0, 0, -0.5))

ILFOCOLOR_400 = ColourNegative(brightness=(0.15, 0, 0))
