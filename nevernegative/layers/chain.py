from typing import Sequence

from torch import Tensor

from nevernegative.layers.base import Layer
from nevernegative.layers.typing import LayerCallable


class LayerChain(Layer):
    def __init__(self, layers: Sequence[Layer | LayerCallable | None]) -> None:
        super().__init__()
        self.layers = [layer for layer in layers if layer is not None]

    def __call__(self, image: Tensor) -> Tensor:
        if self.layers is not None:
            for layer in self.layers:
                image = layer(image)

        return image
