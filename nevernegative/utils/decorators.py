import functools
from typing import TYPE_CHECKING, Callable, Concatenate, ParamSpec, TypeVar

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from nevernegative.layers.base import Layer

P = ParamSpec("P")
LayerT = TypeVar("LayerT", bound="Layer")


def save_figure(
    f: Callable[Concatenate[LayerT, P], Figure],
) -> Callable[Concatenate[LayerT, str, P], None]:
    @functools.wraps(f)
    def wrapper(self: LayerT, name: str, *args: P.args, **kwargs: P.kwargs) -> None:
        config = self.get_setup_config()

        if config.plotting is None:
            return

        figure = f(self, *args, **kwargs)
        figure.set_size_inches(config.plotting.figure_size)

        directory = config.plotting.plot_path / f"{config.layer_index:02}_{self.get_layer_name()}"
        directory.mkdir(parents=True, exist_ok=True)

        figure.savefig(directory / name, format="png")

        plt.close()

    return wrapper
