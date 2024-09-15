from pydantic import BaseModel

from nevernegative.typing.config import ColorBalancerConfig, CropperConfig, DewarperConfig


class ScannerConfig(BaseModel):
    dewarping: DewarperConfig
    cropping: CropperConfig
    color_balancing: ColorBalancerConfig

    callbacks: list
