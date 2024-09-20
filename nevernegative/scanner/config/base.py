from typing import Sequence

from pydantic import BaseModel, Field

from nevernegative.callbacks.base import Callback
from nevernegative.color.base import ColorBalancer
from nevernegative.crop.base import Cropper
from nevernegative.dewarp.base import Dewarper
from nevernegative.typing.config import ColorBalancerConfigs, CropperConfigs, DewarperConfigs


class ScannerConfig(BaseModel):
    dewarping: DewarperConfigs | Dewarper | None = None
    cropping: CropperConfigs | Cropper | None = None
    color_balancing: ColorBalancerConfigs | ColorBalancer | None = None

    callbacks: Sequence[Callback] = Field(default_factory=list)
