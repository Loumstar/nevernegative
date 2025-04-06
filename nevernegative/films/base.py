from typing import Sequence

from nevernegative.layers.base import Layer


class Film:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self._layers = layers

    @property
    def layers(self) -> Sequence[Layer]:
        return self._layers
