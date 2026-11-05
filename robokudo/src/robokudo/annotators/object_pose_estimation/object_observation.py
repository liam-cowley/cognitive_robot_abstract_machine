"""
An object marked by a mask in a color and depth frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from typing_extensions import TYPE_CHECKING, Self

from robokudo.annotators.object_pose_estimation.exceptions import (
    ImageResolutionMismatch,
)
from robokudo.cas import CASViews
from robokudo.types.cv import Point2D, Rect
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image

if TYPE_CHECKING:
    import numpy.typing as numpy_typing
    from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel

    from robokudo.cas import CAS
    from robokudo.types.cv import ColorToDepthRatio, ImageROI


# %% Observations


@dataclass
class ColorAndDepthFrame:
    """
    A color and a depth image of the same resolution, taken by one camera.
    """

    color_image: numpy_typing.NDArray[numpy.uint8]
    """
    Color image in RGB channel order.
    """

    depth_image: numpy_typing.NDArray[numpy.float32]
    """
    Depth image in meters; pixels without a measurement are zero.
    """

    camera_model: PinholeCameraModel
    """
    Calibration of the camera at the resolution of the images.
    """

    @classmethod
    def from_cas(cls, cas: CAS, depth_units_per_meter: float) -> Self:
        """
        :param cas: The CAS holding the images and the camera observation.
        :param depth_units_per_meter: Depth image values per meter.
        :return: The color and depth image of the CAS at depth image resolution.
        """
        color_image = get_scaled_color_image_for_depth_image(
            cas, cas.get(CASViews.COLOR_IMAGE)
        )
        color_to_depth_ratio = cas.color_to_depth_ratio
        return cls(
            color_image=cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB),
            depth_image=cas.get(CASViews.DEPTH_IMAGE).astype(numpy.float32)
            / depth_units_per_meter,
            camera_model=cas.require_camera_observation().effective_camera_model.scaled(
                scale_x=color_to_depth_ratio.horizontal,
                scale_y=color_to_depth_ratio.vertical,
            ),
        )

    def __post_init__(self) -> None:
        """
        :raises ImageResolutionMismatch: If the images differ in resolution.
        """
        color_height, color_width = self.color_image.shape[:2]
        depth_height, depth_width = self.depth_image.shape[:2]
        if (color_height, color_width) != (depth_height, depth_width):
            raise ImageResolutionMismatch(
                color_image_resolution=CameraResolution(
                    width=color_width, height=color_height
                ),
                depth_image_resolution=CameraResolution(
                    width=depth_width, height=depth_height
                ),
            )


@dataclass
class ObjectObservation:
    """
    One object in a frame, marked by a mask.
    """

    frame: ColorAndDepthFrame
    """
    The frame the object is seen in.
    """

    mask: numpy_typing.NDArray[numpy.bool_]
    """
    Pixels of :attr:`frame` that show the object.
    """

    @classmethod
    def from_image_roi(
        cls,
        frame: ColorAndDepthFrame,
        image_roi: ImageROI,
        color_to_depth_ratio: ColorToDepthRatio,
    ) -> Self:
        """
        :param frame: The frame the object is seen in.
        :param image_roi: Region of the object in the color image, with its mask.
        :param color_to_depth_ratio: Size of the depth image relative to the color image.
        :return: The object, with its mask placed over the whole depth image.
        """
        region = color_to_depth_ratio.depth_region(image_roi.roi)
        region_mask = cv2.resize(
            image_roi.mask.astype(numpy.uint8),
            (region.width, region.height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(numpy.bool_)
        mask = numpy.zeros(frame.depth_image.shape[:2], dtype=numpy.bool_)
        region_in_image = mask[
            region.pos.y : region.pos.y + region.height,
            region.pos.x : region.pos.x + region.width,
        ]
        region_in_image[:] = region_mask[
            : region_in_image.shape[0], : region_in_image.shape[1]
        ]
        return cls(frame=frame, mask=mask)

    @cached_property
    def mask_region(self) -> Rect:
        """
        The smallest rectangle containing every pixel of :attr:`mask`.
        """
        left, top, width, height = cv2.boundingRect(self.mask.astype(numpy.uint8))
        return Rect(pos=Point2D(x=left, y=top), width=width, height=height)

    def is_projected_into_mask_region(
        self, point: numpy_typing.NDArray[numpy.float64], margin: float
    ) -> bool:
        """
        :param point: A point in the camera frame.
        :param margin: Fraction of the mask region size added to each side of the region.
        :return: Whether the point lies in front of the camera and projects into the
            enlarged mask region.
        """
        if point[2] <= 0:
            return False
        x, y, _ = self.frame.camera_model.intrinsic_matrix @ (point / point[2])
        region = self.mask_region
        left, top, right, bottom = region.get_corner_points()
        horizontal_margin = region.width * margin
        vertical_margin = region.height * margin
        return (
            left - horizontal_margin <= x <= right + horizontal_margin
            and top - vertical_margin <= y <= bottom + vertical_margin
        )
