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
            raise RuntimeError()

        figure = f(self, *args, **kwargs)

        if self.figure_size is not None:
            figure.set_size_inches(self.figure_size)

        self.plot_path.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.plot_path / name)

        plt.close()

    return wrapper
