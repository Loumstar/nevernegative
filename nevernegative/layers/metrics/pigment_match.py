import torch
from torch import Tensor

from nevernegative.layers.base import Layer


class PigmentMatch(Layer):
    plotting_name = "color_match"

    def __init__(
        self,
        pigment: tuple[int, int, int],
        *,
        stops: float = 0,
        max_stops: float | None = None,
        ord: int = 1,
    ) -> None:
        super().__init__()

        self.pigment = torch.tensor(pigment, dtype=torch.float32) / 255

        self.pigment.unsqueeze_(-1)
        self.pigment.unsqueeze_(-1)

        self.stops = stops
        self.max_stops = max_stops
        self.ord = ord

    def adjusted_pigment(self, stops: float) -> Tensor:
        return (self.pigment * (2**stops)).clamp(0, 1)

    def forward(self, image: Tensor) -> Tensor:
        if self.max_stops is not None:
            image = image.clone()

            intensities = torch.linalg.vector_norm(image, dim=-3)
            max_intensity = torch.linalg.vector_norm(
                self.adjusted_pigment(self.max_stops).to(image.device), dim=-3
            )

            mask = intensities > max_intensity

            image[..., mask] = 0

            if self.plotting:
                self.plot("filtered.png", image)

        distances: Tensor = torch.linalg.vector_norm(
            image - self.adjusted_pigment(self.stops).to(image.device),
            dim=-3,
            keepdim=True,
        )

        return (1 - (distances / distances.max())) ** self.ord
