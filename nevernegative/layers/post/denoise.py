from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from skimage.restoration import denoise_tv_chambolle, estimate_sigma

from nevernegative.layers.base import Layer


class DenoiseChannels(Layer):
    def __init__(
        self,
        weight: float | tuple[float, float, float],
        *,
        plot_path: Path | None = None,
        figure_size: tuple[int, int] = (15, 15),
    ) -> None:
        super().__init__(plot_path, figure_size)

        self.weight = weight if isinstance(weight, tuple) else (weight, weight, weight)

    def __call__(self, image: NDArray) -> NDArray:
        denoised = (image.copy() * 2) - 1

        self.plot("original.png", image)

        print(estimate_sigma(image, channel_axis=-1))

        for channel, weight in enumerate(self.weight):
            if weight > 0:
                denoised[..., channel] = denoise_tv_chambolle(denoised[..., channel], weight=weight)

        result = np.clip((denoised + 1) / 2, 0, 1)

        self.plot("denoised.png", result)

        return result
