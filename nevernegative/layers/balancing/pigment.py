from typing import Literal

import torch
import torchvision.transforms.functional as F
from torch import Tensor

from nevernegative.layers.balancing.base import Balancer
from nevernegative.utils.brightness import compute_pixel_brightness


class RemoveEmulsionPigment(Balancer):
    plotting_name = "pigment"
    supported_films: dict[str, tuple[int, int, int]] = {
        "DELTA_100": (237, 236, 255),
        "CINESTILL_800T": (255, 169, 114),
        "COLOR_PLUS_200": (255, 178, 121),
        "ILFOCOLOR_400": (255, 208, 153),
    }

    def __init__(
        self,
        pigment: tuple[int, int, int]
        | Literal["DELTA_100", "CINESTILL_800T", "COLOR_PLUS_200", "auto"],
        *,
        brightness_correction: bool = True,
        mode: Literal["divide", "gamma"] = "gamma",
    ) -> None:
        super().__init__()

        self.brightness_correction = brightness_correction
        self.mode: Literal["divide", "gamma"] = mode

        if pigment == "auto":
            raise NotImplementedError("Estimating pigment is not yet supported.")

        if isinstance(pigment, str):
            pigment = self.supported_films[pigment]

        self.pigment = torch.tensor(pigment, dtype=torch.float32).reshape((3, 1, 1)) / 255
        self.brightness_factor = compute_pixel_brightness(*self.pigment.squeeze().tolist())

    def forward(self, image: Tensor) -> Tensor:
        if self.brightness_correction:
            image = F.adjust_brightness(image, self.brightness_factor)

        if self.mode == "divide":
            return image / self.pigment.to(image.device)

        [red, green, blue] = self.pigment.squeeze().tolist()

        return torch.stack(
            (
                F.adjust_brightness(image.select(-3, 0), 1 / red),
                F.adjust_brightness(image.select(-3, 1), 1 / green),
                F.adjust_brightness(image.select(-3, 2), 1 / blue),
            ),
            dim=-3,
        )
