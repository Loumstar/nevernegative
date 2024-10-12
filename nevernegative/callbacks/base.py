from abc import ABC

from nevernegative.color.base import ColorBalancer
from nevernegative.crop.base import Cropper
from nevernegative.dewarp.base import Dewarper
from nevernegative.image.image import Image
from nevernegative.layers.base import Layer
from nevernegative.scanner.base import Scanner


class Callback(ABC):
    def on_layer_begin(self, layer: Layer, image: Image) -> None:
        """Callback made before a layer is computed.

        Args:
            layer (Layer): _description_
            image (Image): _description_
        """

    def on_layer_end(self, layer: Layer, image: Image) -> None:
        """Callback made after a layer is computed.

        Args:
            layer (Layer): _description_
            image (Image): _description_
        """

    def on_dewarp_begin(self, dewarper: Dewarper, image: Image) -> None:
        """Callback made before a block is computed.

        Args:
            dewarper (Dewarper): _description_
            image (Image): _description_
        """

    def on_dewarp_end(self, dewarper: Dewarper, image: Image) -> None:
        """Callback made after a layer is computed.

        Args:
            dewarper (Dewarper): _description_
            image (Image): _description_
        """

    def on_crop_begin(self, cropper: Cropper, image: Image) -> None:
        """Callback made before a block is computed.

        Args:
            cropper (Cropper): _description_
            image (Image): _description_
        """

    def on_crop_end(self, cropper: Cropper, image: Image) -> None:
        """Callback made after a layer is computed.

        Args:
            cropper (Cropper): _description_
            image (Image): _description_
        """

    def on_color_balance_begin(self, color_balancer: ColorBalancer, image: Image) -> None:
        """Callback made before a block is computed.

        Args:
            color_balancer (ColorBalancer): _description_
            image (Image): _description_
        """

    def on_color_balance_end(self, color_balancer: ColorBalancer, image: Image) -> None:
        """Callback made after a layer is computed.

        Args:
            color_balancer (ColorBalancer): _description_
            image (Image): _description_
        """

    def on_scan_begin(self, scanner: Scanner, image: Image) -> None:
        """Callback made before a layer is computed.

        Args:
            scanner (Scanner): _description_
            image (Image): _description_
        """

    def on_scan_end(self, scanner: Scanner, image: Image) -> None:
        """Callback made after a layer is computed.

        Args:
            scanner (Scanner): _description_
            image (Image): _description_
        """
