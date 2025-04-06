from nevernegative.films.base import Film
from nevernegative.layers.balancing.brightness import Brightness
from nevernegative.layers.balancing.contrast import Contrast
from nevernegative.layers.balancing.grey import Grey
from nevernegative.layers.balancing.invert import Invert
from nevernegative.layers.balancing.pigment import RemoveEmulsionPigment
from nevernegative.layers.balancing.saturation import Saturation
from nevernegative.layers.balancing.temperature import Temperature
from nevernegative.layers.utils.clip import Clip


class ColorPlus(Film):
    def __init__(self, temperature: int = 5600) -> None:
        super().__init__(
            layers=[
                Temperature(temperature),
                RemoveEmulsionPigment(pigment=(190, 117, 70)),
                Invert(),
                Brightness(0.72, channel=0),
                Brightness(0.6, channel=1),
                Brightness(0.49, channel=2),
                Contrast(2.2),
                Brightness(1.4),
                Saturation(1),
            ]
        )


class Ektar(Film):
    def __init__(self, temperature: int = 5600) -> None:
        super().__init__(
            layers=[
                Temperature(temperature),
                RemoveEmulsionPigment(pigment=(234, 123, 98)),
                Invert(),
                Brightness(1, channel=0),
                Brightness(0.95, channel=1),
                Brightness(0.7, channel=2),
                Brightness(1.4),
                Contrast(2),
            ]
        )


class Gold(Film):
    def __init__(self, temperature: int = 5600) -> None:
        super().__init__(
            layers=[
                Temperature(temperature),
                RemoveEmulsionPigment(pigment=(255, 170, 109)),
                Invert(),
                Brightness(0.82, channel=0),
                Brightness(0.6, channel=1),
                Brightness(0.51, channel=2),
                Contrast(3),
                Brightness(1.2),
                Saturation(1),
            ]
        )


class TriX(Film):
    def __init__(self, temperature: int = 5600) -> None:
        super().__init__(
            layers=[
                Temperature(temperature),
                RemoveEmulsionPigment(pigment=(185, 177, 192)),
                Invert(),
                Grey(channel=2),
                Clip(),
            ]
        )
