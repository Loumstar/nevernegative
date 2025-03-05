from typing import Callable, TypeAlias

from torch import Tensor

LayerCallable: TypeAlias = Callable[[Tensor], Tensor]
