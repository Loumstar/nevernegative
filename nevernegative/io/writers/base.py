from abc import ABC, abstractmethod
from pathlib import Path

from torch import Tensor


class Writer(ABC):
    @abstractmethod
    def save(self, path: Path, name: str, image: Tensor) -> None: ...
