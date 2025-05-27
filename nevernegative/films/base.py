from typing import Sequence

from nevernegative.layers.balancing.gamma import Gamma
from nevernegative.layers.balancing.temperature import Temperature
from nevernegative.layers.base import Layer
from nevernegative.layers.utils.clip import Clip
from nevernegative.layers.utils.rotate import Rotate


class Film:
    _exposure_gamma_multiplier = 1.8
    _static_layers: Sequence[Layer] = ()

    def __init__(
        self,
        *,
        temperature: int | None = None,
        exposure: float = 0,
        rotation: float = 0,
    ) -> None:
        self._layers = list(self._static_layers)

        if temperature is not None:
            self._layers.insert(0, Temperature(temperature, mode="remove"))

        if rotation != 0:
            self._layers.insert(0, Rotate(rotation))

        if exposure != 0:
            self._layers.append(Gamma(self._exposure_gamma_multiplier**-exposure))

        self._layers.append(Clip())

    @property
    def layers(self) -> Sequence[Layer]:
        return self._layers
