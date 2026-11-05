"""
The pipeline step that gives classified object hypotheses a pose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from timeit import default_timer

from krrood.ormatic.utils import classproperty
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.annotators.object_pose_estimation.exceptions import (
    PoseTrackingNotSupported,
)
from robokudo.annotators.object_pose_estimation.object_observation import (
    ColorAndDepthFrame,
    ObjectObservation,
)
from robokudo.annotators.object_pose_estimation.pose_update_mode import (
    EstimateAllObjects,
    EstimatedPose,
    PoseCandidate,
    PoseRequest,
    PoseUpdateMode,
)
from robokudo.types.annotation import Classification, PoseAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.decorators import record_time
from robokudo.utils.error_handling import catch_and_raise_to_blackboard

if TYPE_CHECKING:
    import numpy
    import numpy.typing as numpy_typing
    from krrood_spatial_math.spatial_types import Pose


# %% Object pose estimator


@dataclass(eq=False)
class ObjectPoseEstimator(ThreadedAnnotator, ABC):
    """
    Give classified object hypotheses with masks a pose, in the camera frame.

    Each posed object hypothesis gets a :class:`PoseAnnotation`.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Parameters of an object pose estimator.
        """

        @dataclass
        class Parameters:
            """
            Pose estimation settings.
            """

            pose_update_mode: PoseUpdateMode = field(default_factory=EstimateAllObjects)
            """
            Which objects get a new pose, and how.
            """

            depth_units_per_meter: float = 1000.0
            """
            Depth image values per meter.
            """

        parameters = Parameters()
        """
        Pose estimation defaults.
        """

    name: str = __qualname__
    """
    Name of the behaviour-tree node.
    """

    descriptor: Descriptor = field(default_factory=Descriptor)
    """
    Pose estimation and visualization configuration.
    """

    def __post_init__(self) -> None:
        """
        Initialize the threaded behaviour.

        :raises PoseTrackingNotSupported: If the pose update mode needs tracking this
            annotator cannot do.
        """
        super().__init__(self.name, self.descriptor)
        self.require_supported_pose_update_mode()

    @classproperty
    def supports_pose_tracking(cls) -> bool:
        """
        Whether this annotator can track known poses.
        """
        return False

    def require_supported_pose_update_mode(self) -> None:
        """
        :raises PoseTrackingNotSupported: If the pose update mode needs tracking this
            annotator cannot do.
        """
        pose_update_mode = self.descriptor.parameters.pose_update_mode
        if pose_update_mode.requires_pose_tracking and not self.supports_pose_tracking:
            raise PoseTrackingNotSupported(annotator_class_name=self.get_class_name())

    @abstractmethod
    def estimate_object_pose(self, request: PoseRequest) -> Pose | None:
        """
        Estimate the pose of an object without prior knowledge.

        :param request: The object hypothesis to pose.
        :return: Pose of the object in the camera frame, or ``None`` if none was found.
        """

    def can_pose_class(self, class_name: str) -> bool:
        """
        :param class_name: Name of an object class.
        :return: Whether objects of the class can be posed.
        """
        return True

    def compute_object_center(
        self, request: PoseRequest, pose: Pose
    ) -> numpy_typing.NDArray[numpy.float64]:
        """
        :param request: The object hypothesis.
        :param pose: Pose of the object in the camera frame.
        :return: The center of the object in the camera frame.
        """
        return pose.position.to_np()[:3]

    def determine_object_pose(self, request: PoseRequest) -> Pose | None:
        """
        :param request: The object hypothesis to pose.
        :return: Pose of the object in the camera frame, or ``None`` if none was found.
        """
        return self.estimate_object_pose(request)

    @catch_and_raise_to_blackboard
    @record_time
    def compute(self) -> Status:
        """
        Pose the classified object hypotheses of the CAS and visualize the poses.

        :return: :attr:`Status.SUCCESS` after processing the frame.
        """
        start_time = default_timer()
        self.require_supported_pose_update_mode()
        frame = ColorAndDepthFrame.from_cas(
            self.get_cas(), self.descriptor.parameters.depth_units_per_meter
        )
        requests = self.create_pose_requests(frame)
        estimated_poses = [
            EstimatedPose(request=request, object_pose=object_pose)
            for request in requests
            if (object_pose := self.determine_object_pose(request)) is not None
        ]
        self.annotate_poses(estimated_poses)
        if self.descriptor.parameters.global_with_visualization:
            self.visualize(frame, estimated_poses)
        self.feedback_message = (
            f"Posed {len(estimated_poses)} of {len(requests)} objects in "
            f"{default_timer() - start_time:.4f}s"
        )
        return Status.SUCCESS

    def create_pose_requests(self, frame: ColorAndDepthFrame) -> list[PoseRequest]:
        """
        :param frame: The current frame.
        :return: The requests of the object hypotheses of the CAS that get a pose.
        """
        return [
            request
            for object_hypothesis in self.get_cas().filter_annotations_by_type(
                ObjectHypothesis
            )
            if (request := self.create_pose_request(object_hypothesis, frame))
            is not None
        ]

    def create_pose_request(
        self,
        object_hypothesis: ObjectHypothesis,
        frame: ColorAndDepthFrame,
    ) -> PoseRequest | None:
        """
        Decide whether and how an object hypothesis gets a pose in this frame.

        :param object_hypothesis: The object hypothesis.
        :param frame: The current frame.
        :return: The request to pose the object hypothesis, or ``None`` if it gets no
            pose.
        """
        cas = self.get_cas()
        classifications = cas.filter_by_type(
            Classification, object_hypothesis.annotations
        )
        if object_hypothesis.roi.mask is None or not classifications:
            return None
        classification = max(
            classifications, key=lambda candidate: candidate.confidence
        )
        if not self.can_pose_class(classification.classname):
            return None
        pose_annotations = cas.filter_by_type(
            PoseAnnotation, object_hypothesis.annotations
        )
        return self.descriptor.parameters.pose_update_mode.create_pose_request(
            self.create_pose_candidate(
                object_hypothesis,
                classification.classname,
                ObjectObservation.from_image_roi(
                    frame, object_hypothesis.roi, cas.color_to_depth_ratio
                ),
                pose_annotations[-1] if pose_annotations else None,
            )
        )

    def create_pose_candidate(
        self,
        object_hypothesis: ObjectHypothesis,
        class_name: str,
        object_observation: ObjectObservation,
        pose_annotation: PoseAnnotation | None,
    ) -> PoseCandidate:
        """
        :param object_hypothesis: The object hypothesis.
        :param class_name: Name of its class.
        :param object_observation: The object hypothesis in the current frame.
        :param pose_annotation: The latest pose annotation it holds.
        :return: What the pose update mode decides on.
        """
        return PoseCandidate(
            object_hypothesis=object_hypothesis,
            class_name=class_name,
            object_observation=object_observation,
            pose_annotation=pose_annotation,
        )

    def annotate_poses(self, estimated_poses: list[EstimatedPose]) -> None:
        """
        Write the estimated poses of this frame into pose annotations.

        :param estimated_poses: The poses of this frame.
        """
        for estimated_pose in estimated_poses:
            self.annotate_pose(estimated_pose)

    def annotate_pose(self, estimated_pose: EstimatedPose) -> None:
        """
        Write an estimated pose into the pose annotation of its request, adding one if
        the request has none.

        :param estimated_pose: The estimated pose.
        """
        request = estimated_pose.request
        pose_annotation = request.pose_annotation
        if pose_annotation is None:
            pose_annotation = PoseAnnotation()
            request.object_hypothesis.annotations.append(pose_annotation)
        pose_annotation.source = self.get_class_name()
        pose_annotation.translation = estimated_pose.object_pose.position.to_np()[
            :3
        ].tolist()
        pose_annotation.rotation = (
            estimated_pose.object_pose.orientation.to_np().tolist()
        )

    def visualize(
        self, frame: ColorAndDepthFrame, estimated_poses: list[EstimatedPose]
    ) -> None:
        """
        Show the posed objects; this base shows nothing of its own.

        .. note::
            :class:`~robokudo.annotators.object_pose_visualizer.ObjectPoseVisualizer` shows
            the pose annotations of any object hypothesis.

        :param frame: The current frame.
        :param estimated_poses: The poses of this frame.
        """
