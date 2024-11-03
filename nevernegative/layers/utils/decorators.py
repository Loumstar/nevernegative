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
        if self.plot_path is None:
            return

        figure = f(self, *args, **kwargs)
        path = self.plot_path / name

        path.parent.mkdir(parents=True, exist_ok=True)
        figure.set_size_inches(self.figure_size)

        figure.savefig(path)

        plt.close()

    return wrapper
