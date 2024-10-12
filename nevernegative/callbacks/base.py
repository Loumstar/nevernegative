from abc import ABC

from numpy.typing import NDArray

from nevernegative.layers.base import Layer


class Callback(ABC):
    def on_layer_begin(self, layer: Layer, image: NDArray) -> None:
        """Callback made before a layer is computed.

        Args:
            layer (Layer): _description_
            image (NDArray): _description_
        """

    def on_layer_end(self, layer: Layer, image: NDArray) -> None:
        """Callback made after a layer is computed.

        Args:
            layer (Layer): _description_
            image (NDArray): _description_
        """
