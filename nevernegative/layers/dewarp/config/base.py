from abc import ABC
from typing import TypeVar

from nevernegative.layers.config import LayerConfig
from nevernegative.layers.dewarp.base import Dewarper

DewarperT = TypeVar("DewarperT", bound=Dewarper)


class DewarperConfig(LayerConfig[DewarperT], ABC):
    pass
