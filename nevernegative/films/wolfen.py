from nevernegative.films.base import Film
from nevernegative.layers.balancing.brightness import Brightness
from nevernegative.layers.balancing.invert import Invert
from nevernegative.layers.balancing.pigment import RemoveEmulsionPigment


class WolfenNC500(Film):
    _static_layers = [
        RemoveEmulsionPigment(pigment=(165, 132, 88)),
        Invert(),
        Brightness(1.1, channel=0),
    ]
