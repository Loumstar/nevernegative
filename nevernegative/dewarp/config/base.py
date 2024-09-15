from abc import ABC

from pydantic import BaseModel


class DewarperConfig(BaseModel, ABC):
    type: str
