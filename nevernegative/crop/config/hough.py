import numpy as np
from pydantic import BaseModel

from nevernegative.crop.config.base import CropperConfig
from nevernegative.crop.hough import HoughCrop
from nevernegative.layers.config.edge import EdgeDetectConfig
from nevernegative.layers.config.grey import GreyConfig
from nevernegative.layers.config.threshold import ThresholdConfig


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
            preprocessing_layers=self.initialize_preprocessing_layers(),
        )
