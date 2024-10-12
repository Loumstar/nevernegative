from abc import ABC, abstractmethod
from typing import TypeVar

from nevernegative.layers.color.base import ColorBalancer
from nevernegative.layers.config import LayerConfig

ColorBalancerT = TypeVar("ColorBalancerT", bound=ColorBalancer)


class ColorBalancerConfig(LayerConfig[ColorBalancerT], ABC):
    type: str

    @abstractmethod
    def initialize(self) -> ColorBalancerT:
        """Initialize the cropper class.

        Returns:
            Cropper: _description_
        """
