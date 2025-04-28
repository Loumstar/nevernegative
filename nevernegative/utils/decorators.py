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
        if self._debug_config is None:
            return

        figure = f(self, *args, **kwargs)
        figure.set_size_inches(self._debug_config.figure_size)

        self._debug_config.plot_path.mkdir(parents=True, exist_ok=True)
        figure.savefig(self._debug_config.plot_path / name, format="png")

        plt.close()

    return wrapper
