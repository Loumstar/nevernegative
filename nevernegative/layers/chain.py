from typing import Any, Sequence

from numpy.typing import NDArray

from nevernegative.layers.base import Layer
from nevernegative.layers.typing import LayerCallable


class LayerChain(Layer):
    def __init__(self, layers: Sequence[Layer | LayerCallable] | None) -> None:
        super().__init__()
        self.layers = layers

    def __call__(self, image: NDArray[Any]) -> NDArray[Any]:
        if self.layers is not None:
            for layer in self.layers:
                image = layer(image)

        return image
