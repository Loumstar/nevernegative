from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar

from pydantic import BaseModel, Field

from nevernegative.dewarp.base import Dewarper
from nevernegative.layers.base import Layer
from nevernegative.layers.config.base import LayerConfig
from nevernegative.typing.config import LayerConfigs

DewarperT = TypeVar("DewarperT", bound=Dewarper)


class DewarperConfig(BaseModel, Generic[DewarperT], ABC):
    type: str

    preprocessing_layers: Sequence[LayerConfigs] | Sequence[Layer] = Field(default_factory=list)

    def initialize_preprocessing_layers(self) -> Sequence[Layer]:
        return [
            layer.initialize() if isinstance(layer, LayerConfig) else layer
            for layer in self.preprocessing_layers
        ]

    @abstractmethod
    def initialize(self) -> DewarperT:
        """Initialize the cropper class.

        Returns:
            Cropper: _description_
        """
