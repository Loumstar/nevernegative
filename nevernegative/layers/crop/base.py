from abc import abstractmethod

import torch
from torch import Tensor

from nevernegative.layers.base import Layer


class Cropper(Layer):
    def _get_corners(self, coords: Tensor, image_shape: tuple[int, ...]) -> Tensor:
        # Create a copy shifted into the negative w-scale to find TR/BL values
        shifted_coords = torch.stack((coords, coords), dim=-2)
        shifted_coords[..., 1, 1] -= image_shape[-1]

        distance = torch.linalg.vector_norm(shifted_coords, dim=-1)  # n x 2

        corners = torch.cat(
            (
                coords[distance.argmin(dim=0)],  # TL, TR
                coords[distance.argmax(dim=0)],  # BR, BL
            ),
        )

        return corners

    @abstractmethod
    def forward(self, image: Tensor) -> Tensor:
        """Crop an image, returning only the portion that contains the negative.

        Args:
            image (RGBImage[DTypeT]): _description_

        Returns:
            RGBImage[DTypeT]: _description_
        """
