import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import torch
import tqdm
from torch import Tensor

from nevernegative.io.readers.base import Reader
from nevernegative.io.writers.base import Writer
from nevernegative.layers.base import Layer, PlotConfig
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
        super().__init__(
            layers,
            reader=reader,
            writer=writer,
            device=device,
        )

        self._plotting_config: PlotConfig | None = None

    @contextmanager
    def plot(
        self,
        plot_path: Path,
        figure_size: tuple[int, int] = (15, 15),
    ) -> Iterator[None]:
        try:
            self._plotting_config = PlotConfig(
                plot_path=plot_path,
                figure_size=figure_size,
            )

            yield None

        finally:
            self._plotting_config = None

    def _process_image(self, image: Tensor, *, image_path: Path | None = None) -> Tensor:
        for index, layer in enumerate(self.layers):
            with layer.setup(
                image_path,
                index,
                plotting=self._plotting_config,
            ):
                image = layer(image)

        return image

    def array(self, image: Tensor) -> Tensor:
        return self._process_image(image)

    def file(self, source: Path, destination: Path) -> Tensor:
        image = self.reader.load(source, self.device)
        output = self._process_image(image, image_path=source)

        self.writer.save(destination, source.name, output)

        return output

    def glob(
        self,
        directory: Path,
        destination: Path,
        *,
        glob: str = "*",
    ) -> None:
        files = sorted(directory.glob(glob))

        if not files:
            raise RuntimeError("No images found.")

        for path in tqdm.tqdm(files, desc="Proccesing images"):
            self.file(path, destination)
