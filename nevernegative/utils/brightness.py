from typing import Literal, TypeVar

from torch import Tensor

T = TypeVar("T", float, Tensor)


def compute_pixel_brightness(red: T, green: T, blue: T) -> T:
    return (0.3 * red) + (0.59 * green) + (0.11 * blue)


def compute_image_brightness(
    image: Tensor,
    *,
    mode: Literal["mean", "brightness"] = "brightness",
    keepdim: bool = True,
    channel_axis: int = -3,
) -> Tensor:
    channel_first = image.swapaxes(0, channel_axis)

    if mode == "brightness":
        [red, green, blue] = channel_first.unsqueeze(1)
        brightness = compute_pixel_brightness(red, green, blue)
    else:
        brightness = channel_first.mean(0, keepdim=True)

    brightness.swapaxes_(0, channel_axis)

    return brightness if keepdim else brightness.squeeze(channel_axis)
