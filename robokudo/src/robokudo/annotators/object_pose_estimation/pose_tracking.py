"""
The capability of an object pose estimator to track known poses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from krrood.ormatic.utils import classproperty
from krrood_spatial_math.spatial_types import Pose
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.object_pose_estimation.object_pose_estimator import (
    ObjectPoseEstimator,
)
from robokudo.annotators.object_pose_estimation.pose_update_mode import (
    EstimatedPose,
    PoseCandidate,
    PoseRequest,
    PoseUpdateMode,
)
from robokudo.annotators.object_pose_estimation.previous_frame_poses import (
    PosedObject,
    PreviousFramePoses,
)

if TYPE_CHECKING:
    from robokudo.annotators.object_pose_estimation.object_observation import (
        ObjectObservation,
    )
    from robokudo.types.annotation import PoseAnnotation
    from robokudo.types.scene import ObjectHypothesis


# %% Tracking requests


@dataclass(eq=False)
class TrackingPoseRequest(PoseRequest):
    """
    A request to track a known pose instead of estimating one.
    """

    previous_pose: Pose = field(kw_only=True)
    """
    Known pose of the object in the camera frame to track from.
    """


@dataclass(eq=False)
class TrackingPoseCandidate(PoseCandidate):
    """
    A pose candidate that also knows the objects posed in the previous frame.
    """

    previous_frame_poses: PreviousFramePoses = field(kw_only=True)
    """
    Objects of the previous frame not yet matched.
    """

    def to_tracking_request(
        self, previous_pose: Pose, pose_annotation: PoseAnnotation | None = None
    ) -> TrackingPoseRequest:
        """
        :param previous_pose: Pose to track from.
        :param pose_annotation: Annotation receiving the new pose, or ``None`` to add one.
        :return: The request to track the object hypothesis.
        """
        return TrackingPoseRequest(
            object_hypothesis=self.object_hypothesis,
            class_name=self.class_name,
            object_observation=self.object_observation,
            pose_annotation=pose_annotation,
            previous_pose=previous_pose,
        )


# %% Tracking pose update modes


@dataclass
class TrackingPoseUpdateMode(PoseUpdateMode, ABC):
    """
    A mode that tracks objects with a known pose: one from their :class:`PoseAnnotation`
    or the pose of the matching object of the previous frame.
    """

    @classproperty
    def requires_pose_tracking(cls) -> bool:
        """
        Like :attr:`PoseUpdateMode.requires_pose_tracking`, but these modes track.
        """
        return True

    update_existing_pose_annotations: bool = False
    """
    Whether a tracked pose overwrites the :class:`PoseAnnotation` it was tracked from
    instead of being added as a new one.
    """

    track_across_frames: bool = True
    """
    Whether objects without a :class:`PoseAnnotation` are tracked from the matching object
    posed in the previous frame.
    """

    minimum_track_overlap: float = 0.1
    """
    Intersection over union of the mask regions below which an object does not match an
    object of the previous frame.
    """

    def create_annotation_tracking_request(
        self, candidate: TrackingPoseCandidate
    ) -> TrackingPoseRequest:
        """
        :param candidate: An object hypothesis holding a pose annotation.
        :return: The request to track it from that annotation.
        """
        pose_annotation = candidate.pose_annotation
        return candidate.to_tracking_request(
            previous_pose=Pose.from_xyz_quaternion(
                *pose_annotation.translation, *pose_annotation.rotation
            ),
            pose_annotation=(
                pose_annotation if self.update_existing_pose_annotations else None
            ),
        )

    def create_previous_frame_tracking_request(
        self, candidate: TrackingPoseCandidate
    ) -> PoseRequest | None:
        """
        :param candidate: An object hypothesis without a pose annotation.
        :return: The request to track it from the matching object of the previous frame,
            or ``None`` if there is none to track from.
        """
        if not self.track_across_frames:
            return None
        previous_pose = candidate.previous_frame_poses.take_match(
            class_name=candidate.class_name,
            mask_region=candidate.object_observation.mask_region,
            minimum_overlap=self.minimum_track_overlap,
        )
        if previous_pose is None:
            return None
        return candidate.to_tracking_request(previous_pose=previous_pose)


@dataclass
class EstimateAndTrack(TrackingPoseUpdateMode):
    """
    Objects with a known pose are tracked, the others are estimated.
    """

    def create_pose_request(
        self, candidate: TrackingPoseCandidate
    ) -> PoseRequest | None:
        if candidate.pose_annotation is not None:
            return self.create_annotation_tracking_request(candidate)
        return (
            self.create_previous_frame_tracking_request(candidate)
            or candidate.to_request()
        )


@dataclass
class TrackPosedObjects(TrackingPoseUpdateMode):
    """
    Only objects with a known pose are tracked.
    """

    def create_pose_request(
        self, candidate: TrackingPoseCandidate
    ) -> PoseRequest | None:
        if candidate.pose_annotation is not None:
            return self.create_annotation_tracking_request(candidate)
        return self.create_previous_frame_tracking_request(candidate)


# %% Pose tracking


@dataclass(eq=False)
class PoseTracking(ObjectPoseEstimator, ABC):
    """
    An object pose estimator that tracks known poses, and estimates a pose anew when its
    tracked pose is not plausible.
    """

    class Descriptor(ObjectPoseEstimator.Descriptor):
        """
        Parameters of a tracking object pose estimator.
        """

        @dataclass
        class Parameters(ObjectPoseEstimator.Descriptor.Parameters):
            """
            Pose estimation and tracking settings.
            """

            maximum_track_center_offset: float = 0.25
            """
            Fraction of the mask region size by which the center of a tracked object may
            lie outside the mask region before the pose is estimated anew.
            """

        parameters = Parameters()
        """
        Pose estimation and tracking defaults.
        """

    descriptor: Descriptor = field(default_factory=Descriptor)
    """
    Pose estimation and tracking configuration.
    """

    previous_frame_poses: PreviousFramePoses = field(
        init=False, default_factory=PreviousFramePoses, repr=False
    )
    """
    Objects posed in the previous frame that are not yet matched in this one.
    """

    @classproperty
    def supports_pose_tracking(cls) -> bool:
        """
        Like :attr:`ObjectPoseEstimator.supports_pose_tracking`, but this annotator tracks.
        """
        return True

    @abstractmethod
    def track_object_pose(self, request: TrackingPoseRequest) -> Pose:
        """
        Refine the previous pose of a request to the current frame.

        :param request: The object hypothesis to track, with a previous pose.
        :return: The refined pose of the object in the camera frame.
        """

    def determine_object_pose(self, request: PoseRequest) -> Pose | None:
        """
        Like :meth:`ObjectPoseEstimator.determine_object_pose`, but track a known pose
        if the tracked pose is plausible, and estimate it otherwise.
        """
        if not isinstance(request, TrackingPoseRequest):
            return super().determine_object_pose(request)
        tracked_pose = self.track_object_pose(request)
        if self.is_plausible_track(request, tracked_pose):
            return tracked_pose
        self.rk_logger.debug(
            f"Tracked pose of {request.class_name} left its mask, estimating it anew."
        )
        return self.estimate_object_pose(request)

    def create_pose_candidate(
        self,
        object_hypothesis: ObjectHypothesis,
        class_name: str,
        object_observation: ObjectObservation,
        pose_annotation: PoseAnnotation | None,
    ) -> TrackingPoseCandidate:
        """
        Like :meth:`ObjectPoseEstimator.create_pose_candidate`, with the objects of the
        previous frame.
        """
        return TrackingPoseCandidate(
            object_hypothesis=object_hypothesis,
            class_name=class_name,
            object_observation=object_observation,
            pose_annotation=pose_annotation,
            previous_frame_poses=self.previous_frame_poses,
        )

    def annotate_poses(self, estimated_poses: list[EstimatedPose]) -> None:
        """
        Like :meth:`ObjectPoseEstimator.annotate_poses`, and keep the posed objects for
        matching in the next frame.
        """
        super().annotate_poses(estimated_poses)
        self.previous_frame_poses = PreviousFramePoses(
            unmatched_objects=[
                PosedObject(
                    class_name=estimated_pose.request.class_name,
                    pose=estimated_pose.object_pose,
                    mask_region=estimated_pose.request.object_observation.mask_region,
                )
                for estimated_pose in estimated_poses
            ]
        )

    def is_plausible_track(self, request: PoseRequest, tracked_pose: Pose) -> bool:
        """
        :param request: The tracked object hypothesis.
        :param tracked_pose: Tracked pose of the object in the camera frame.
        :return: Whether the center of the object lies in front of the camera and projects
            into the mask region, enlarged by the maximum track center offset.
        """
        return request.object_observation.is_projected_into_mask_region(
            self.compute_object_center(request, tracked_pose),
            self.descriptor.parameters.maximum_track_center_offset,
        )
