from pathlib import Path

from torch import Tensor
from torchvision.utils import save_image

from nevernegative.io.writers.base import Writer


class TorchWriter(Writer):
    def __init__(self, extension: str = "png") -> None:
        self.extension = extension

    def save(self, path: Path, name: str, image: Tensor) -> None:
        save_image(image, (path / name).with_suffix(f".{self.extension}"))
