import glob
import logging
from pathlib import Path

import rawpy
import skimage as ski
import tqdm
from numpy.typing import NDArray

from nevernegative.scanner.base import Scanner

LOGGER = logging.getLogger(__name__)


class SimpleScanner(Scanner):
    def _from_file(self, source: str | Path, *, is_raw: bool = False) -> NDArray:
        if isinstance(source, str):
            source = Path(source)

        if is_raw:
            with rawpy.imread(str(source)) as raw:
                image = raw.postprocess().copy()
        else:
            image = ski.io.imread(source)

        return ski.util.img_as_float64(image)

    def array(self, image: NDArray) -> NDArray:
        for layer in (self.dewarper, self.cropper, self.color_balancer):
            if layer is None:
                continue

            image = layer(image)

        return image

    def file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        is_raw: bool = False,
    ) -> NDArray:
        """Transform the image from a file.

        Args:
            source (str | Path): _description_
            destination (str | Path): _description_
            return_array (bool, optional): _description_. Defaults to False.
            is_raw (bool, optional): _description_. Defaults to False.
            target_layer (str | int | None, optional): _description_. Defaults to None.

        Returns:
            NDArray[Any] | None: _description_
        """
        image = self._from_file(source, is_raw=is_raw)
        output = self.array(image)

        if isinstance(source, str):
            source = Path(source)

        if isinstance(destination, str):
            destination = Path(destination)

        destination.mkdir(parents=True, exist_ok=True)
        result = ski.util.img_as_ubyte(output)

        ski.io.imsave(destination / source.with_suffix(".png").name, result)

        return output

    def glob(self, source: str, destination: str | Path, *, is_raw: bool = False) -> None:
        files = glob.glob(source)
        files.sort()

        if not files:
            raise RuntimeError("No images found.")

        for file in tqdm.tqdm(files, desc="Proccesing images"):
            try:
                self.file(file, destination, is_raw=is_raw)
            except Exception:
                LOGGER.exception(f"Failed to process {file}")
