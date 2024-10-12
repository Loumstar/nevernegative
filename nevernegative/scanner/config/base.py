from typing import Sequence

from pydantic import BaseModel, Field

from nevernegative.callbacks.base import Callback
from nevernegative.layers.base import Layer


class ScannerConfig(BaseModel):
    layers: Sequence[Layer] = Field(min_length=1)
    callbacks: Sequence[Callback] = Field(default_factory=list)
