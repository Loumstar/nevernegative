import numpy as np
import skimage as ski
from numpy.typing import NDArray
from pydantic import BaseModel

from nevernegative.layers.color.base import ColorBalancer


class ChannelPreset(BaseModel):
    exposure: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0


class PresetColorBalancer(ColorBalancer):
    def __init__(
        self,
        exposure: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        red: ChannelPreset = ChannelPreset(),
        green: ChannelPreset = ChannelPreset(),
        blue: ChannelPreset = ChannelPreset(),
    ) -> None:
        self.exposure = exposure
        self.brightness = brightness
        self.contrast = contrast

        self.red = red
        self.blue = blue
        self.green = green

    def adjust_brightness(self, image: NDArray) -> NDArray:
        brightness = np.stack([self.red.brightness, self.green.brightness, self.blue.brightness])
        brightness += self.brightness

        return image + brightness

    def adjust_contrast(self, image: NDArray) -> NDArray:
        contrast = np.stack([self.red.contrast, self.green.contrast, self.blue.contrast])
        contrast += self.contrast

        return ((image - 0.5) / (1 + contrast)) + 0.5

    # def adjust_exposure(self, image: NDArray) -> NDArray:
    #     exposure = np.stack([self.red.exposure, self.green.exposure, self.blue.exposure])
    #     exposure += self.exposure

    #     return image * (1 + exposure)

    def __call__(self, image: NDArray) -> NDArray:
        image = ski.util.invert(image)

        image = self.adjust_contrast(image)
        image = self.adjust_brightness(image)

        return np.clip(image, 0, 1)
