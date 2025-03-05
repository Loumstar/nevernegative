from pathlib import Path

from nevernegative.layers.balancing.brightness import Brightness
from nevernegative.layers.balancing.contrast import Contrast
from nevernegative.layers.balancing.invert import Invert
from nevernegative.layers.balancing.pigment import RemoveEmulsionPigment
from nevernegative.layers.balancing.saturation import Saturation
from nevernegative.layers.balancing.temperature import Temperature
from nevernegative.scanner.simple import SimpleScanner

scanner = SimpleScanner(
    device="mps",
    layers=[
        Temperature(temperature=5200),
        RemoveEmulsionPigment(pigment="COLOR_PLUS_200"),
        Invert(),
        Brightness(0.78, channel=0),
        Brightness(0.6, channel=1),
        Brightness(0.5, channel=2),
        Contrast(2.5),
        Brightness(1.0),
        Saturation(1),
    ],
)

image_folder = "/Users/louismanestar/Documents/Projects/Film Scanner/nevernegative/images/Brighton"
batch_name = "results/notebook"

scanner.glob(
    source=(Path(image_folder) / "*.NEF").as_posix(),
    destination=Path(image_folder) / batch_name,
    is_raw=True,
)
