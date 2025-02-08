from pydantic import BaseModel


class Film(BaseModel):
    brightness: float | tuple[float, float, float] = 0.0
    contrast: float | tuple[float, float, float] = 0.0
    saturation: float = 0.0

    is_negative: bool
    is_colour: bool

    pigment: tuple[int, int, int] | None = None
    grey_channel: int | None = None
