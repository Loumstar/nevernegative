from typing import TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

ScalarTypeT = TypeVar("ScalarTypeT", bound=np.floating)

Image: TypeAlias = npt.NDArray[ScalarTypeT]
ThresholdImage: TypeAlias = npt.NDArray[np.bool]
EdgeMap: TypeAlias = npt.NDArray[np.bool]
