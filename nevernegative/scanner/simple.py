import glob
import logging
from pathlib import Path

import tqdm
from torch import Tensor
from torchvision.utils import save_image

from nevernegative.scanner.base import Scanner

LOGGER = logging.getLogger(__name__)


class SimpleScanner(Scanner):
    def array(self, image: Tensor) -> Tensor:
        for layer in self.layers:
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
    ) -> Tensor:
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

        save_image(output, destination / source.with_suffix(".png").name)

        return output

    def glob(
        self,
        source: str,
        destination: str | Path,
        *,
        is_raw: bool = False,
        raise_exceptions: bool = False,
    ) -> None:
        files = glob.glob(source)
        files.sort()

        if not files:
            raise RuntimeError("No images found.")

        for file in tqdm.tqdm(files, desc="Proccesing images"):
            try:
                self.file(file, destination, is_raw=is_raw)

            except Exception as e:
                if raise_exceptions:
                    raise e

                LOGGER.exception(f"Failed to process {file}")
