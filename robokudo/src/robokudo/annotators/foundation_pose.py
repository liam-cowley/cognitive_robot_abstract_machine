"""
6D pose estimation and tracking of classified object hypotheses with FoundationPose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

from robokudo.annotators.object_pose_estimation.mesh_based_pose_estimation import (
    MeshBasedPoseEstimation,
)
from robokudo.annotators.object_pose_estimation.pose_tracking import (
    EstimateAndTrack,
    PoseTracking,
)

if TYPE_CHECKING:
    from pathlib import Path

    from krrood_spatial_math.spatial_types import Pose

    from robokudo.annotators.foundation_pose_inference import FoundationPoseInference
    from robokudo.annotators.object_pose_estimation.pose_tracking import (
        TrackingPoseRequest,
    )
    from robokudo.annotators.object_pose_estimation.pose_update_mode import (
        PoseRequest,
        PoseUpdateMode,
    )


# %% FoundationPose annotator


@dataclass(eq=False)
class FoundationPoseAnnotator(MeshBasedPoseEstimation, PoseTracking):
    """
    Estimate and track the 6D pose of classified object hypotheses with masks, using the
    mesh of their class and the FoundationPose model.
    """

    class Descriptor(MeshBasedPoseEstimation.Descriptor, PoseTracking.Descriptor):
        """
        Parameters of the FoundationPose annotator.
        """

        @dataclass
        class Parameters(
            MeshBasedPoseEstimation.Descriptor.Parameters,
            PoseTracking.Descriptor.Parameters,
        ):
            """
            Pose estimation, tracking and visualization settings.
            """

            pose_update_mode: PoseUpdateMode = field(default_factory=EstimateAndTrack)
            """
            Which objects get a new pose, and how.
            """

        parameters = Parameters()
        """
        Pose estimation and visualization defaults.
        """

    foundation_pose_inference: FoundationPoseInference = field(kw_only=True)
    """
    The model estimating and tracking the poses.
    """

    name: str = __qualname__
    """
    Name of the behaviour-tree node.
    """

    descriptor: Descriptor = field(default_factory=Descriptor)
    """
    Pose estimation and visualization configuration.
    """

    def load_mesh_files(self, mesh_files: list[Path]) -> None:
        self.foundation_pose_inference.load_mesh_files(mesh_files)

    def select_mesh_file(self, mesh_file: Path) -> None:
        self.foundation_pose_inference.select_mesh_file(mesh_file)

    def estimate_object_pose(self, request: PoseRequest) -> Pose | None:
        return self.foundation_pose_inference.estimate_pose(request.object_observation)

    def track_object_pose(self, request: TrackingPoseRequest) -> Pose:
        return self.foundation_pose_inference.track_pose(
            request.object_observation, request.previous_pose
        )
