import skimage as ski
import torch
from torch import Tensor

from nevernegative.layers.base import Layer


class EdgeDetect(Layer):
    plotting_name = "edge_detect"

    def __init__(
        self,
        *,
        sigma: float = 1,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
    ) -> None:
        super().__init__()

        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def forward(self, image: Tensor) -> Tensor:
        edge = ski.feature.canny(
            image.squeeze().cpu().numpy(),
            sigma=self.sigma,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )

        return torch.tensor(edge, dtype=torch.float32, device=image.device).unsqueeze(-3)
