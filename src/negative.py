from pathlib import Path
from typing import Any

import rawpy
import skimage as ski
from typing_extensions import Self

from src.layers.crop.blur import GaussianBlur
from src.layers.crop.resize import Resize
from src.layers.crop.threshold import Threshold
from src.typing.image import Image


class Negative:
    def __init__(self, image: Image[Any, Any]) -> None:
        self.image = image

    @classmethod
    def from_file(cls, source: str | Path, *, is_raw: bool = False) -> Self:
        if is_raw:
            with rawpy.imread(source) as raw:
                image = raw.postprocess().copy()
        else:
            image = ski.io.imread(source)

        return cls(image)

    def run(self):
        if self.image.shape[-1] == 4:
            image = ski.color.rgba2rgb(self.image)
        else:
            image = self.image

        resized = Resize(height=800)(image)
        grey = ski.color.rgb2gray(resized)
        blur = GaussianBlur(3)(grey)
        threshold = Threshold()(blur)
