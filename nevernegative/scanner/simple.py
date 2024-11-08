import glob
from pathlib import Path
from typing import Literal, overload

import numpy as np
import rawpy
import skimage as ski
import tqdm
from numpy.typing import NDArray

from nevernegative.scanner.base import Scanner


class SimpleScanner(Scanner):
    def _load_from_file(self, source: str | Path, *, is_raw: bool = False) -> NDArray:
        if isinstance(source, str):
            source = Path(source)

        if is_raw:
            with rawpy.imread(str(source)) as raw:
                image = raw.postprocess().copy()
        else:
            image = ski.io.imread(source)

        return image

    def array(self, image: NDArray) -> NDArray:
        for layer in (self.dewarper, self.cropper, self.color_balancer):
            if layer is None:
                continue

            image = layer(image)

        return image

    @overload
    def file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        return_array: Literal[True] = True,
        is_raw: bool = False,
    ) -> NDArray: ...

    @overload
    def file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        return_array: Literal[False] = False,
        is_raw: bool = False,
    ) -> None: ...

    def file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        return_array: bool = False,
        is_raw: bool = False,
    ) -> NDArray | None:
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
        image = self._load_from_file(source, is_raw=is_raw)
        output = self.array(image)

        if isinstance(source, str):
            source = Path(source)

        if isinstance(destination, str):
            destination = Path(destination)

        destination.mkdir(parents=True, exist_ok=True)

        if not output.dtype == np.uint8:
            output = (255 * output).astype(np.uint8)

        ski.io.imsave(destination / source.with_suffix(".png").name, output)

        if return_array:
            return output

        return None

    def glob(self, source: str, destination: str | Path, *, is_raw: bool = False) -> None:
        files = glob.glob(source)

        if not files:
            raise RuntimeError("No images found.")

        for file in tqdm.tqdm(files, desc="Proccesing images"):
            self.file(file, destination, is_raw=is_raw)
