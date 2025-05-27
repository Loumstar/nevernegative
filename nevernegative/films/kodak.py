from nevernegative.films.base import Film
from nevernegative.layers.balancing.brightness import Brightness
from nevernegative.layers.balancing.contrast import Contrast
from nevernegative.layers.balancing.gamma import Gamma
from nevernegative.layers.balancing.grey import Grey
from nevernegative.layers.balancing.invert import Invert
from nevernegative.layers.balancing.pigment import RemoveEmulsionPigment
from nevernegative.layers.balancing.saturation import Saturation


class ColorPlus(Film):
    _static_layers = [
        RemoveEmulsionPigment(pigment=(192, 119, 73)),
        Invert(),
        Brightness(1.1, channel=0),
        Brightness(0.9, channel=2),
        Gamma(2),
        Contrast(1.6),
    ]


class Ultramax(Film):
    _static_layers = [
        RemoveEmulsionPigment(pigment=(192, 119, 73)),
        Invert(),
        Brightness(1.1, channel=0),
        Brightness(0.9, channel=2),
        Gamma(2),
        Contrast(1.6),
    ]


class Ektar(Film):
    _static_layers = [
        RemoveEmulsionPigment(pigment=(234, 123, 98)),
        Invert(),
        Brightness(0.82, channel=2),
        Contrast(2),
    ]


class Gold(Film):
    _static_layers = [
        RemoveEmulsionPigment(pigment=(255, 170, 109)),
        Invert(),
        Brightness(0.82, channel=0),
        Brightness(0.6, channel=1),
        Brightness(0.51, channel=2),
        Contrast(3),
        Brightness(1.2),
        Saturation(1),
    ]


class TriX(Film):
    _static_layers = [
        RemoveEmulsionPigment(pigment=(185, 177, 192)),
        Invert(),
        Grey(channel=2),
    ]


class Portra800(Film):
    _static_layers = [
        RemoveEmulsionPigment(pigment=(174, 102, 58)),
        Invert(),
        Brightness(1.1, channel=0),
    ]
