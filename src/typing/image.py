from enum import Enum, auto
from typing import Any, Literal, TypeAlias, TypeVar

from numpy import bool, dtype, floating, ndarray

DTypeT = TypeVar("DTypeT", bound=dtype[Any])
FloatingT = TypeVar("FloatingT", bound=dtype[floating])

ShapeT = TypeVar("ShapeT", bound=tuple[int, ...])

GreyShape: TypeAlias = tuple[int, int]
ColorShape: TypeAlias = tuple[int, int, Literal[3]]

Image: TypeAlias = ndarray[ShapeT, DTypeT]

ColorImage: TypeAlias = Image[tuple[int, int, Literal[3]], DTypeT]
GreyImage: TypeAlias = Image[GreyShape, DTypeT]
ThresholdImage: TypeAlias = Image[GreyShape, dtype[bool]]
EdgeMap: TypeAlias = Image[GreyShape, dtype[bool]]


class ImageType(Enum):
    COLOUR = auto()
    GREY = auto()
    THRESHOLD = auto()
    EDGE = auto()
    CROPPER = auto()
