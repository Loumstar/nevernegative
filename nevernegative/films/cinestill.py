from nevernegative.films.base import Film
from nevernegative.layers.balancing.brightness import Brightness
from nevernegative.layers.balancing.contrast import Contrast
from nevernegative.layers.balancing.invert import Invert
from nevernegative.layers.balancing.pigment import RemoveEmulsionPigment


class CineStill800T(Film):
    _static_layers = [
        RemoveEmulsionPigment(pigment=(190, 115, 69)),
        Invert(),
        Brightness(1, channel=0),
        Brightness(0.90, channel=1),
        Brightness(0.65, channel=2),
        Brightness(1.4),
        Contrast(2),
    ]
