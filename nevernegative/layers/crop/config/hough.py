import numpy as np
from pydantic import BaseModel

from nevernegative.layers.common.config.edge import EdgeDetectConfig
from nevernegative.layers.common.config.grey import GreyConfig
from nevernegative.layers.common.config.threshold import ThresholdConfig
from nevernegative.layers.crop.config.base import CropperConfig
from nevernegative.layers.crop.hough import HoughCrop


class HoughTransformParameters(BaseModel):
    start_angle: float = np.deg2rad(-45)
    end_angle: float = np.deg2rad(135)
    step: int = 360


class HoughCropConfig(CropperConfig[HoughCrop]):
    peak_ratio: float = 0.2
    peak_min_distance: int = 30
    snap_to_edge_map: bool = True

    grey_converter: GreyConfig
    thresholder: ThresholdConfig
    edge_detector: EdgeDetectConfig

    hough_transform_parameters: HoughTransformParameters

    def initialize(self) -> HoughCrop:
        return HoughCrop(
            peak_ratio=self.peak_ratio,
            min_distance=self.peak_min_distance,
            snap_to_edge_map=self.snap_to_edge_map,
            grey_converter=self.grey_converter,
            thresholder=self.thresholder,
            edge_detector=self.edge_detector,
            hough_transform_parameters=self.hough_transform_parameters,
        )
