from pathlib import Path

from numpy.typing import NDArray

from nevernegative.layers.balancing.base import Balancer
from nevernegative.layers.balancing.presets.film.base import Film
from nevernegative.layers.chain import LayerChain
from nevernegative.layers.common.brightness import Brightness
from nevernegative.layers.common.clip import Clip
from nevernegative.layers.common.contrast import Contrast
from nevernegative.layers.common.grey import Grey
from nevernegative.layers.common.positive import Positive
from nevernegative.layers.common.saturation import Saturation


class BasicBalance(Balancer):
    def __init__(
        self,
        film: Film,
        *,
        clip: bool = True,
        invert: bool = True,
        plot_path: Path | None = None,
        figure_size: tuple[int, int] = (15, 15),
    ) -> None:
        super().__init__(plot_path=plot_path, figure_size=figure_size)
        self.film = film

        self.invert = invert

        self.make_positive = Positive(film.is_negative and invert)
        self.postprocess = LayerChain(
            (
                Contrast(film.contrast),
                Brightness(film.brightness),
                Saturation(film.saturation),
                Clip() if clip else None,
            )
        )

        self.make_black_and_white = Grey(film.grey_channel)

    def __call__(self, image: NDArray) -> NDArray:
        positive = self.make_positive(image)
        result = self.postprocess(positive)

        if not self.film.is_colour:
            result = self.make_black_and_white(result)

        self.plot_balancing("balanced.png", result)

        return result
