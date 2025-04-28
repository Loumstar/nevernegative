from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import Tensor


class Reader(ABC):
    @abstractmethod
    def load(self, path: Path, device: str | torch.device = "cpu") -> Tensor: ...
