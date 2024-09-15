from pathlib import Path

import rawpy
import skimage as ski

from nevernegative.scanner.base import Scanner
from nevernegative.typing.image import Image


class SimpleScanner(Scanner):
    def from_file(self, source: str | Path, *, is_raw: bool = False) -> Image:
        if is_raw:
            with rawpy.imread(source) as raw:
                image = raw.postprocess().copy()
        else:
            image = ski.io.imread(source)

        return image

    # def run(self):
    #     if self.image.shape[-1] == 4:
    #         image = ski.color.rgba2rgb(self.image)
    #     else:
    #         image = self.image

    #     resized = Resize(height=800)(image)
    #     grey = ski.color.rgb2gray(resized)
    #     blur = GaussianBlur(3)(grey)
    #     threshold = Threshold()(blur)
