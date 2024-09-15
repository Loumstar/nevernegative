from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from nevernegative.layers.base import Layer

LayerT = TypeVar("LayerT", bound=Layer)


class LayerConfig(BaseModel, Generic[LayerT], ABC):
    type: str

    @abstractmethod
    def initialize(self) -> LayerT:
        """Initialize layer class.

        Returns:
            LayerT: _description_
        """
