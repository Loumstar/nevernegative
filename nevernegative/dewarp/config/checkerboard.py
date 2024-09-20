from typing import Literal

from nevernegative.dewarp.checkerboard import CheckerboardDewarper
from nevernegative.dewarp.config.base import DewarperConfig


class CheckerboardDewarperConfig(DewarperConfig[CheckerboardDewarper]):
    type: Literal["checkerboard"] = "checkerboard"

    def initialize(self) -> CheckerboardDewarper:
        return CheckerboardDewarper(
            preprocessing_layers=self.initialize_preprocessing_layers(),  # type: ignore[abstract]
        )
