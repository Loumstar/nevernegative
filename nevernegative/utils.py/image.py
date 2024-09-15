from typing import TypeGuard

from nevernegative.typing.image import DTypeT, GreyImage, Image, RGBImage


def is_grey_image(image: Image[DTypeT]) -> TypeGuard[GreyImage[DTypeT]]:
    return image.ndim == 2


def is_rgb_image(image: Image[DTypeT]) -> TypeGuard[RGBImage[DTypeT]]:
    return image.ndim == 3 and image.shape[-1] == 3
