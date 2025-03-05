import skimage as ski
from torch import Tensor

from nevernegative.layers.base import Layer


class EstimateNoise(Layer):
    plotting_name = "estimate_noise"

    def forward(self, image: Tensor) -> Tensor:
        if self._is_bw(image):
            sigma = ski.restoration.estimate_sigma(image.permute(1, 2, 0).cpu().numpy())
            print(f"Greyscale noise: {sigma:.5e}")

        else:
            print("Color noise:")
            for i, name in enumerate(("red", "green", "blue")):
                sigma = ski.restoration.estimate_sigma(image.select(-3, i).cpu().numpy())
                print(f"\t{name.capitalize()} noise: {sigma:.3e}")

        return image
