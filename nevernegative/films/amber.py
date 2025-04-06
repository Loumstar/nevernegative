from nevernegative.films.base import Film
from nevernegative.layers.balancing.brightness import Brightness
from nevernegative.layers.balancing.contrast import Contrast
from nevernegative.layers.balancing.invert import Invert
from nevernegative.layers.balancing.pigment import RemoveEmulsionPigment
from nevernegative.layers.balancing.temperature import Temperature


class Amber800T(Film):
    def __init__(self, temperature: int = 5600) -> None:
        super().__init__(
            layers=[
                Temperature(temperature),
                RemoveEmulsionPigment(pigment=(222, 121, 78)),
                Invert(),
                Brightness(1, channel=0),
                Brightness(0.90, channel=1),
                Brightness(0.65, channel=2),
                Brightness(1.4),
                Contrast(2),
            ]
        )
