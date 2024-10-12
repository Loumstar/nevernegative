from typing import Literal

from nevernegative.layers.dewarp.config.base import DewarperConfig
from nevernegative.layers.dewarp.hough import HoughTransformDewarper


class HoughTransformDewarperConfig(DewarperConfig[HoughTransformDewarper]):
    type: Literal["hough"] = "hough"
