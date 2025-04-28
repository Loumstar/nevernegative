from pathlib import Path
from typing import Literal, TypeAlias

import kornia as K
import rawpy
import torch
from torch import Tensor

from nevernegative.io.readers.base import Reader

SupportedDemosaicAlgorithm: TypeAlias = Literal[
    "aahd",
    "afd",
    "ahd",
    "amaze",
    "dcb",
    "dht",
    "linear",
    "lmmse",
    "modified_ahd",
    "ppg",
    "vcd",
    "vcd_modified_ahd",
    "vng",
]

SupportedColorSpace: TypeAlias = Literal[
    "aces",
    "adobe",
    "p3d65",
    "prophoto",
    "rec2020",
    "wide",
    "xyz",
    "raw",
    "srgb",
]

FBDDNoiseReduction: TypeAlias = Literal["off", "light", "full"]

HighlightMode: TypeAlias = Literal["ignore", "clip", "blend"]

WhiteBalanceMultipliers: TypeAlias = tuple[float, float, float, float]


class RawPyReader(Reader):
    demosaic_algorithm_name_mapping: dict[SupportedDemosaicAlgorithm, str] = {
        "aahd": "AAHD",
        "afd": "AFD",
        "ahd": "AHD",
        "amaze": "AMAZE",
        "dcb": "DCB",
        "dht": "DHT",
        "linear": "LINEAR",
        "lmmse": "LMMSE",
        "modified_ahd": "MODIFIED_AHD",
        "ppg": "PPG",
        "vcd": "VCD",
        "vcd_modified_ahd": "VCD_MODIFIED_AHD",
        "vng": "VNG",
    }

    color_space_name_mapping: dict[SupportedColorSpace, str] = {
        "aces": "ACES",
        "adobe": "Adobe",
        "p3d65": "P3D65",
        "prophoto": "ProPhoto",
        "rec2020": "Rec2020",
        "wide": "Wide",
        "xyz": "XYZ",
        "raw": "raw",
        "srgb": "sRGB",
    }

    fbdd_noise_reduction_name_mapping: dict[FBDDNoiseReduction, str] = {
        "off": "Off",
        "light": "Light",
        "full": "Full",
    }

    orientation_mapping: dict[int, int] = {
        -90: 6,
        0: 0,
        90: 5,
        180: 3,
    }

    highlight_name_mapping: dict[HighlightMode, str] = {
        "ignore": "Ignore",
        "clip": "Clip",
        "blend": "Blend",
    }

    def __init__(
        self,
        demosaic_algorithm: SupportedDemosaicAlgorithm = "ahd",
        half_size: bool = False,
        four_color_rgb: bool = False,
        dcb_iterations: int = 0,
        dcb_enhance: bool = False,
        fbdd_noise_reduction: FBDDNoiseReduction = "off",
        noise_threshold: float | None = None,
        median_filter_passes: int = 0,
        white_balance: Literal["camera", "auto"] | WhiteBalanceMultipliers = (0, 0, 0, 0),
        color_space: SupportedColorSpace = "srgb",
        bits_per_channel: Literal[8, 16] = 8,
        orientation: Literal[-90, 0, 90, 180] = 0,
        brightness: float = 1,
        saturation: int | None = None,
        black_level: int | None = None,
        disable_auto_scaling: bool = False,
        disable_auto_brightness: bool = False,
        auto_brightness_threshold: float = 0.01,
        maximum_threshold: float = 0.75,
        highlight_mode: HighlightMode = "clip",
        exposure_shift: int = 0,
        exposure_preserve_highlights: float = 0,
        gamma: tuple[float, float] = (2.222, 4.5),
        chromatic_aberration: tuple[float, float] = (1, 1),
    ) -> None:
        self.demosaic_algorithm = self.demosaic_algorithm_name_mapping[demosaic_algorithm]
        self.fbdd_noise_reduction = self.fbdd_noise_reduction_name_mapping[fbdd_noise_reduction]
        self.color_space = self.color_space_name_mapping[color_space]

        self.half_size = half_size
        self.four_color_rgb = four_color_rgb

        self.dcb_iterations = dcb_iterations
        self.dcb_enhance = dcb_enhance

        self.noise_threshold = noise_threshold
        self.median_filter_passes = median_filter_passes

        self.white_balance: Literal["camera", "auto"] | WhiteBalanceMultipliers = white_balance

        self.bits_per_channel = bits_per_channel
        self.orientation = self.orientation_mapping[orientation]

        self.brightness = brightness
        self.black_level = black_level
        self.saturation = saturation

        self.disable_auto_scaling = disable_auto_scaling
        self.disable_auto_brightness = disable_auto_brightness

        self.auto_brightness_threshold = auto_brightness_threshold
        self.maximum_threshold = maximum_threshold

        self.highlight_mode = self.highlight_name_mapping[highlight_mode]

        self.exposure_shift = 2**exposure_shift
        self.exposure_preserve_highlights = exposure_preserve_highlights

        self.gamma = gamma
        self.chromatic_aberration = chromatic_aberration

    @property
    def _use_auto_wb(self) -> bool:
        return self.white_balance == "auto"

    @property
    def _use_camera_wb(self) -> bool:
        return self.white_balance == "camera"

    @property
    def _user_wb(self) -> list[float] | None:
        return list(self.white_balance) if isinstance(self.white_balance, tuple) else None

    def load(self, path: Path, device: str | torch.device = "cpu") -> Tensor:
        with rawpy.imread(path.as_posix()) as raw:
            image = raw.postprocess(
                demosaic_algorithm=rawpy.DemosaicAlgorithm[self.demosaic_algorithm],  # type: ignore
                half_size=self.half_size,
                four_color_rgb=self.four_color_rgb,
                dcb_iterations=self.dcb_iterations,
                dcb_enhance=self.dcb_enhance,
                fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode[self.fbdd_noise_reduction],  # type: ignore
                noise_thr=self.noise_threshold,
                median_filter_passes=self.median_filter_passes,
                use_camera_wb=self._use_camera_wb,
                use_auto_wb=self._use_auto_wb,
                user_wb=self._user_wb,
                output_color=rawpy.ColorSpace[self.color_space],  # type: ignore
                output_bps=self.bits_per_channel,
                user_flip=self.orientation,
                user_black=self.black_level,
                user_sat=self.saturation,
                no_auto_scale=self.disable_auto_scaling,
                no_auto_bright=self.disable_auto_brightness,
                auto_bright_thr=self.auto_brightness_threshold,
                adjust_maximum_thr=self.maximum_threshold,
                bright=self.brightness,
                highlight_mode=rawpy.HighlightMode[self.highlight_mode],  # type: ignore
                exp_shift=self.exposure_shift,
                exp_preserve_highlights=self.exposure_preserve_highlights,
                gamma=self.gamma,
                chromatic_aberration=self.chromatic_aberration,
            )

        return K.utils.image_to_tensor(image).to(device, torch.float16) / 255
