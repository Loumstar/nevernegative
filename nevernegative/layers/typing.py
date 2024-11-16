from typing import Callable, TypeAlias

from numpy.typing import NDArray

LayerCallable: TypeAlias = Callable[[NDArray], NDArray]
