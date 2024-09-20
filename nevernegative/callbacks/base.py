from abc import ABC, abstractmethod

from nevernegative.color.base import ColorBalancer
from nevernegative.crop.base import Cropper
from nevernegative.dewarp.base import Dewarper
from nevernegative.layers.base import Layer
from nevernegative.scanner.base import Scanner
from nevernegative.typing.image import Image, ScalarTypeT


class Callback(ABC):
    @abstractmethod
    def on_layer_begin(self, layer: Layer, image: Image[ScalarTypeT]) -> None:
        """Callback made before a layer is computed.

        Args:
            layer (Layer): _description_
            image (Image[ScalarTypeT]): _description_
        """

    @abstractmethod
    def on_layer_end(self, layer: Layer, image: Image[ScalarTypeT]) -> None:
        """Callback made after a layer is computed.

        Args:
            layer (Layer): _description_
            image (Image[ScalarTypeT]): _description_
        """

    @abstractmethod
    def on_block_begin(
        self,
        block: Dewarper | Cropper | ColorBalancer,
        image: Image[ScalarTypeT],
    ) -> None:
        """Callback made before a block is computed.

        Args:
            block (Dewarper | Cropper | ColorBalancer): _description_
            image (Image[ScalarTypeT]): _description_
        """

    @abstractmethod
    def on_block_end(
        self,
        block: Dewarper | Cropper | ColorBalancer,
        image: Image[ScalarTypeT],
    ) -> None:
        """Callback made after a layer is computed.

        Args:
            block (Dewarper | Cropper | ColorBalancer): _description_
            image (Image[ScalarTypeT]): _description_
        """

    @abstractmethod
    def on_scan_begin(self, scanner: Scanner, image: Image[ScalarTypeT]) -> None:
        """Callback made before a layer is computed.

        Args:
            scanner (Scanner): _description_
            image (Image[ScalarTypeT]): _description_
        """

    @abstractmethod
    def on_scan_end(self, scanner: Scanner, image: Image[ScalarTypeT]) -> None:
        """Callback made after a layer is computed.

        Args:
            scanner (Scanner): _description_
            image (Image[ScalarTypeT]): _description_
        """
