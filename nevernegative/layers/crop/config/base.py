from abc import ABC
from typing import TypeVar

from nevernegative.layers.config import LayerConfig
from nevernegative.layers.crop.base import Cropper

CropperT = TypeVar("CropperT", bound=Cropper)


class CropperConfig(LayerConfig[CropperT], ABC):
    type: str
