"""
Exceptions raised by object pose estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.datastructures.camera_resolution import CameraResolution

from robokudo.exceptions import RoboKudoError

# %% Object pose estimation


@dataclass
class DuplicateObjectClassMesh(RoboKudoError, ValueError):
    """
    Raised when more than one mesh is configured for an object class.
    """

    class_name: str
    """
    The object class with more than one mesh.
    """

    def error_message(self) -> str:
        return f"More than one mesh is configured for the class '{self.class_name}'."

    def suggest_correction(self) -> str:
        return "Configure exactly one mesh per class name."


@dataclass
class ImageResolutionMismatch(RoboKudoError, ValueError):
    """
    Raised when the color and the depth image of a frame differ in resolution.
    """

    color_image_resolution: CameraResolution
    """
    Resolution of the color image.
    """

    depth_image_resolution: CameraResolution
    """
    Resolution of the depth image.
    """

    def error_message(self) -> str:
        return (
            f"The color image has the resolution {self.color_image_resolution}, the "
            f"depth image {self.depth_image_resolution}."
        )

    def suggest_correction(self) -> str:
        return "Scale the color image to the resolution of the depth image."


@dataclass
class PoseTrackingNotSupported(RoboKudoError, ValueError):
    """
    Raised when a pose update mode asks an annotator without tracking to track a pose.
    """

    annotator_class_name: str
    """
    Class name of the annotator that cannot track.
    """

    def error_message(self) -> str:
        return f"{self.annotator_class_name} cannot track poses."

    def suggest_correction(self) -> str:
        return (
            "Use a pose update mode that only estimates, or an annotator that inherits "
            "PoseTracking."
        )
