from pathlib import Path

import torch
from skimage.restoration import denoise_tv_chambolle
from torch import Tensor

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

    def __call__(self, image: Tensor) -> Tensor:
        denoised = (image * 2) - 1

        self.plot("original.png", image)

        for channel, weight in enumerate(self.weight):
            if weight == 0:
                continue

            result = denoise_tv_chambolle(denoised[..., channel].cpu().numpy(), weight=weight)
            denoised[..., channel] = torch.tensor(result, dtype=denoised.dtype)

        denoised.clamp_(0, 1)

        self.plot("denoised.png", denoised)

        return denoised
