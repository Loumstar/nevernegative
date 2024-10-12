from typing import Literal

from nevernegative.layers.dewarp.checkerboard import CheckerboardDewarper
from nevernegative.layers.dewarp.config.base import DewarperConfig


class CheckerboardDewarperConfig(DewarperConfig[CheckerboardDewarper]):
    type: Literal["checkerboard"] = "checkerboard"
