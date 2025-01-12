from pathlib import Path

from numpy.typing import NDArray
from skimage.restoration import denoise_tv_chambolle

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
        denoised = image.copy()

        self.plot("original.png", image)

        for channel, weight in enumerate(self.weight):
            if weight > 0:
                denoised[..., channel] = denoise_tv_chambolle(image[..., channel], weight=weight)

        self.plot("denoised.png", denoised)

        return denoised
