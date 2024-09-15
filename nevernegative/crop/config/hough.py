import numpy as np
from pydantic import BaseModel

from nevernegative.crop.config.base import CropperConfig
from nevernegative.crop.hough import HoughCrop
from nevernegative.layers.config.edge import EdgeDetectConfig
from nevernegative.layers.config.grey import GreyConfig
from nevernegative.layers.config.threshold import ThresholdConfig


class HoughTransformConfig(BaseModel):
    start_angle: float = np.deg2rad(-45)
    end_angle: float = np.deg2rad(135)
    step: int = 360


class HoughCropConfig(CropperConfig[HoughCrop]):
    peak_ratio: float = 0.2
    peak_min_distance: int = 30
    snap_to_edge_map: bool = True

    grey: GreyConfig
    edge_detector: EdgeDetectConfig
    thresholder: ThresholdConfig

    hough_transform: HoughTransformConfig

    def initialize(self) -> HoughCrop:
        return HoughCrop(
            peak_ratio=self.peak_ratio,
            min_distance=self.peak_min_distance,
            snap_to_edge_map=self.snap_to_edge_map,
            edge_detector=self.edge_detector,
            thresholder=self.thresholder,
            hough_transform=self.hough_transform,
            layers=self.initialize_layers(),
        )
