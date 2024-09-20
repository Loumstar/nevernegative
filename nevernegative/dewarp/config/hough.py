from typing import Literal

from nevernegative.dewarp.config.base import DewarperConfig
from nevernegative.dewarp.hough import HoughTransformDewarper


class HoughTransformDewarperConfig(DewarperConfig[HoughTransformDewarper]):
    type: Literal["hough"] = "hough"

    def initialize(self) -> HoughTransformDewarper:
        return HoughTransformDewarper(
            preprocessing_layers=self.initialize_preprocessing_layers(),  # type: ignore[abstract]
        )
