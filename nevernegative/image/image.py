from pathlib import Path

from numpy.typing import NDArray
from pydantic import BaseModel

from nevernegative.typing.blocks import BlockType


class Image(BaseModel):
    source: Path

    block: BlockType
    layer: str | None = None

    raw: NDArray
