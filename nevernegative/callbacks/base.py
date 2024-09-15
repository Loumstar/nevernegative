from abc import ABC, abstractmethod

from nevernegative.layers.base import Layer
from nevernegative.typing.image import Image, ScalarTypeT


class Callback(ABC):
    @abstractmethod
    def on_layer_begin(self, layer: Layer, image: Image[ScalarTypeT]) -> None:
        """Callback madw before a layer is computed.

        Args:
            layer (Layer): _description_
            image (Image[ScalarTypeT]): _description_
        """

    @abstractmethod
    def on_layer_end(self, layer: Layer, image: Image[ScalarTypeT]) -> None:
        """Callback madw after a layer is computed.

        Args:
            layer (Layer): _description_
            image (Image[ScalarTypeT]): _description_
        """
