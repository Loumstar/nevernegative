from abc import ABC

from pydantic import BaseModel


class ColorBalancerConfig(BaseModel, ABC):
    type: str
