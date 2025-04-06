from nevernegative.films.base import Film
from nevernegative.layers.balancing.brightness import Brightness
from nevernegative.layers.balancing.contrast import Contrast
from nevernegative.layers.balancing.grey import Grey
from nevernegative.layers.balancing.invert import Invert
from nevernegative.layers.balancing.pigment import RemoveEmulsionPigment
from nevernegative.layers.balancing.temperature import Temperature
from nevernegative.layers.utils.clip import Clip


class Delta100(Film):
    def __init__(self, temperature: int = 5600) -> None:
        super().__init__(
            layers=[
                Temperature(temperature),
                RemoveEmulsionPigment(pigment=(236, 237, 255)),
                Invert(),
                Grey(channel=2),
                Brightness(1.2),
                Contrast(1.2),
                Clip(),
            ]
        )


class Delta3200(Film):
    def __init__(self, temperature: int = 5600) -> None:
        super().__init__(
            layers=[
                Temperature(temperature),
                RemoveEmulsionPigment(pigment=(187, 187, 196)),
                Invert(),
                Grey(channel=2),
                Clip(),
            ]
        )


class HP5Plus(Film):
    def __init__(self, temperature: int = 5600) -> None:
        super().__init__(
            layers=[
                Temperature(temperature),
                RemoveEmulsionPigment(pigment=(157, 156, 170)),
                Invert(),
                Grey(channel=2),
                Brightness(1.2),
                Clip(),
            ]
        )


class Ilfocolor(Film):
    def __init__(self, temperature: int = 5600) -> None:
        super().__init__(
            layers=[
                Temperature(temperature),
                RemoveEmulsionPigment(pigment=(255, 208, 153)),
                Invert(),
                Brightness(0.85, channel=0),
                Brightness(0.6, channel=1),
                Brightness(0.5, channel=2),
                Brightness(1.8),
                Contrast(2),
            ]
        )
