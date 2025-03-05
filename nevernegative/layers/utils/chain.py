from typing import Sequence

from torch import Tensor

from nevernegative.layers.base import Layer


class Chain(Layer):
    plotting_name = "chain"

    def __init__(self, layers: Sequence[Layer]) -> None:
        super().__init__()

        self.layers = layers

    def forward(self, image: Tensor) -> Tensor:
        for layer in self.layers:
            plot_path = self.plot_path / layer.plotting_name if self.plot_path is not None else None

            with layer.setup(plot_path, self.figure_size):
                image = layer(image)

        return image
