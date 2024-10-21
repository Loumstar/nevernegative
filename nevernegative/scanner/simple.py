from pathlib import Path
from typing import Literal, overload

import numpy as np
import rawpy
import skimage as ski
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

    @overload
    def file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        return_array: Literal[True],
        is_raw: bool = False,
    ) -> NDArray: ...

    @overload
    def file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        return_array: Literal[False],
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
        # output = Resize(width=800)(output)

        if isinstance(destination, str):
            destination = Path(destination)

        if destination.is_dir():
            destination = destination / "output.png"

        destination.parent.mkdir(parents=True, exist_ok=True)

        if not output.dtype == np.uint8:
            output = (255 * output).astype(np.uint8)

        ski.io.imsave(destination, output)

        if return_array:
            return output

        return None

    def array(self, image: NDArray) -> NDArray:
        for layer in (self.dewarper, self.cropper, self.color_balancer):
            if layer is None:
                continue

            image = layer(image)

        return image
