import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal, Sequence

import torch
import tqdm
from torch import Tensor
from torchvision.utils import save_image

from nevernegative.io.readers.base import Reader
from nevernegative.io.writers.base import Writer
from nevernegative.layers.base import DebugConfig, Layer
from nevernegative.scanner.base import Scanner

LOGGER = logging.getLogger(__name__)


class SimpleScanner(Scanner):
    def __init__(
        self,
        layers: Sequence[Layer],
        *,
        reader: Reader | None = None,
        writer: Writer | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(layers, reader=reader, writer=writer, device=device)

        self._debug_config: DebugConfig | None = None

    @contextmanager
    def debug(self, plot_path: Path, figure_size: tuple[int, int] = (15, 15)) -> Iterator[None]:
        try:
            self._debug_config = DebugConfig(
                plot_path=plot_path,
                figure_size=figure_size,
            )

            yield None

        finally:
            self._debug_config = None

    def _process_image(self, image: Tensor, *, image_path: Path | None = None) -> Tensor:
        for index, layer in enumerate(self.layers):
            with layer.setup(image_path, index, debug=self._debug_config):
                image = layer(image)

        return image

    def array(self, image: Tensor) -> Tensor:
        return self._process_image(image)

    def file(
        self,
        source: Path,
        destination: Path,
        *,
        plot_path: Path | Literal["image_location"] | None = None,
        figure_size: tuple[int, int] = (15, 15),
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

        if plot_path == "image_location":
            destination /= source.with_suffix("").name
            plot_path = destination

        destination.mkdir(parents=True, exist_ok=True)

        output = self.array(image)
        save_image(output, destination / source.with_suffix(".png").name)

        return output

    def glob(
        self,
        directory: Path,
        destination: Path,
        *,
        glob: str = "*",
        plot_path: Path | Literal["image_location"] | None = None,
        figure_size: tuple[int, int] = (15, 15),
    ) -> None:
        files = sorted(directory.glob(glob))

        if not files:
            raise RuntimeError("No images found.")

        for path in tqdm.tqdm(files, desc="Proccesing images"):
            self.file(
                path,
                destination,
                plot_path=plot_path,
                figure_size=figure_size,
            )
