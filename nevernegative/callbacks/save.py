from pathlib import Path
from typing import Literal

import skimage as ski

from nevernegative.callbacks.base import Callback
from nevernegative.image.image import Image
from nevernegative.layers.base import Layer


class SaveImageCallback(Callback):
    def __init__(self, save_dir: str | Path, suffix: Literal[".png", ".jpeg"]) -> None:
        self.save_dir = Path(save_dir)
        self.suffix = suffix

    def on_layer_end(self, layer: Layer, image: Image) -> None:
        image_name = image.source.with_suffix("").name

        path = self.save_dir / image_name / image.block

        if image.layer is not None:
            path = path / image.layer

        ski.io.imsave(path.with_suffix(self.suffix), image.raw)
