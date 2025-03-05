import glob
import logging
from pathlib import Path
from typing import Literal, Sequence

import tqdm
from torch import Tensor, device
from torchvision.utils import save_image

from nevernegative.layers.base import Layer
from nevernegative.scanner.base import Scanner

LOGGER = logging.getLogger(__name__)


class SimpleScanner(Scanner):
    def __init__(self, layers: Sequence[Layer], device: str | device = "cpu") -> None:
        super().__init__(layers, device)

    def array(
        self,
        image: Tensor,
        *,
        plot_path: Path | None = None,
        figure_size: tuple[int, int] = (15, 15),
    ) -> Tensor:
        for i, layer in enumerate(self.layers):
            layer_folder = f"{i:0>2}_{layer.plotting_name}"
            layer_plot_path = plot_path / layer_folder if plot_path is not None else None

            with layer.setup(layer_plot_path, figure_size):
                image = layer(image)

        return image

    def file(
        self,
        source: Path,
        destination: Path,
        *,
        is_raw: bool = False,
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

        output = self.array(image, plot_path=plot_path, figure_size=figure_size)
        save_image(output, destination / source.with_suffix(".png").name)

        return output

    def glob(
        self,
        source: str,
        destination: Path,
        *,
        is_raw: bool = False,
        plot_path: Path | Literal["image_location"] | None = None,
        figure_size: tuple[int, int] = (15, 15),
    ) -> None:
        files = glob.glob(source)
        files.sort()

        if not files:
            raise RuntimeError("No images found.")

        for file in tqdm.tqdm(files, desc="Proccesing images"):
            self.file(
                Path(file),
                destination,
                is_raw=is_raw,
                plot_path=plot_path,
                figure_size=figure_size,
            )
