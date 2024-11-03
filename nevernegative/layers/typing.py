from typing import Callable, TypeAlias

from numpy.typing import NDArray

LayerCallableT: TypeAlias = Callable[[NDArray], NDArray]
