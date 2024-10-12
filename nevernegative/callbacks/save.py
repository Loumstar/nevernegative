from pathlib import Path
from typing import Literal

from numpy.typing import NDArray

from nevernegative.callbacks.base import Callback
from nevernegative.layers.base import Layer


class SaveImageCallback(Callback):
    def __init__(self, save_dir: str | Path, suffix: Literal[".png", ".jpeg"]) -> None:
        self.save_dir = Path(save_dir)
        self.suffix = suffix

    def on_layer_end(self, layer: Layer, image: NDArray) -> None: ...
