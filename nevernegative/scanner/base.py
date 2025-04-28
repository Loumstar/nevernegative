from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from nevernegative.io.readers.base import Reader
from nevernegative.io.readers.rawpy import RawPyReader
from nevernegative.io.writers.base import Writer
from nevernegative.io.writers.torch import TorchWriter
from nevernegative.layers.base import Layer


class Scanner(ABC):
    def __init__(
        self,
        layers: Sequence[Layer],
        *,
        reader: Reader | None = None,
        writer: Writer | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.layers = layers

        self.reader = reader or RawPyReader()
        self.writer = writer or TorchWriter()
        self.device = torch.device(device)

    @abstractmethod
    def array(self, image: Tensor) -> Tensor: ...

    @abstractmethod
    def file(self, source: Path, destination: Path) -> Tensor: ...

    @abstractmethod
    def glob(self, directory: Path, destination: Path, *, glob: str = "*") -> None: ...
