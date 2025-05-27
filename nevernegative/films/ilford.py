from nevernegative.films.base import Film
from nevernegative.layers.balancing.brightness import Brightness
from nevernegative.layers.balancing.contrast import Contrast
from nevernegative.layers.balancing.grey import Grey
from nevernegative.layers.balancing.invert import Invert
from nevernegative.layers.balancing.pigment import RemoveEmulsionPigment


class Delta100(Film):
    _static_layers = [
        RemoveEmulsionPigment(pigment=(236, 237, 255)),
        Invert(),
        Grey(channel=2),
        Brightness(1.2),
        Contrast(1.2),
    ]


class Delta3200(Film):
    _static_layers = [
        RemoveEmulsionPigment(pigment=(187, 187, 196)),
        Invert(),
        Grey(channel=2),
    ]


class HP5Plus(Film):
    _static_layers = [
        RemoveEmulsionPigment(pigment=(157, 156, 170)),
        Invert(),
        Grey(channel=2),
        Brightness(1.2),
    ]


class Ilfocolor(Film):
    _static_layers = [
        RemoveEmulsionPigment(pigment=(255, 208, 153)),
        Invert(),
        Brightness(0.85, channel=0),
        Brightness(0.6, channel=1),
        Brightness(0.5, channel=2),
        Brightness(1.8),
        Contrast(2),
    ]


class Kentmere100(Film):
    _static_layers = [
        RemoveEmulsionPigment(pigment=(167, 164, 180)),
        Invert(),
        Grey(channel=2),
        Brightness(1.2),
        Contrast(1.2),
    ]


class XP2(Film):
    _static_layers = [
        RemoveEmulsionPigment(pigment=(174, 147, 186)),
        Invert(),
        Grey(channel=2),
        Contrast(1.2),
    ]
