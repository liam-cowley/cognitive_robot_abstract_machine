"""
The decision which object hypotheses get a new pose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from krrood.ormatic.utils import classproperty
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from krrood_spatial_math.spatial_types import Pose

    from robokudo.annotators.object_pose_estimation.object_observation import (
        ObjectObservation,
    )
    from robokudo.types.annotation import PoseAnnotation
    from robokudo.types.scene import ObjectHypothesis


# %% Pose requests


@dataclass(eq=False)
class PoseRequest:
    """
    An object hypothesis whose pose is determined in the current frame.
    """

    object_hypothesis: ObjectHypothesis
    """
    The object hypothesis to pose.
    """

    class_name: str
    """
    Name of the class of :attr:`object_hypothesis`.
    """

    object_observation: ObjectObservation
    """
    The object hypothesis in the current frame.
    """

    pose_annotation: PoseAnnotation | None = None
    """
    Annotation of :attr:`object_hypothesis` that receives the new pose, or ``None`` to add
    a new annotation.
    """


@dataclass(eq=False)
class EstimatedPose:
    """
    The pose determined for a :class:`PoseRequest`.
    """

    request: PoseRequest
    """
    The request the pose was determined for.
    """

    object_pose: Pose
    """
    Pose of the object in the camera frame.
    """

    @property
    def label(self) -> str:
        """
        Text identifying the posed object in visualizations.
        """
        return f"{self.request.object_hypothesis.id}: {self.request.class_name}"


@dataclass(eq=False)
class PoseCandidate:
    """
    An object hypothesis that can be posed, before deciding whether and how.
    """

    object_hypothesis: ObjectHypothesis
    """
    The object hypothesis.
    """

    class_name: str
    """
    Name of the class of :attr:`object_hypothesis`.
    """

    object_observation: ObjectObservation
    """
    The object hypothesis in the current frame.
    """

    pose_annotation: PoseAnnotation | None
    """
    The latest pose annotation the object hypothesis holds, in the camera frame.
    """

    def to_request(self) -> PoseRequest:
        """
        :return: The request to estimate the pose of the object hypothesis.
        """
        return PoseRequest(
            object_hypothesis=self.object_hypothesis,
            class_name=self.class_name,
            object_observation=self.object_observation,
        )


# %% Pose update modes


@dataclass
class PoseUpdateMode(ABC):
    """
    Decides which objects get a new pose, and how.
    """

    @classproperty
    def requires_pose_tracking(cls) -> bool:
        """
        Whether only an annotator that can track poses can use this mode.
        """
        return False

    @abstractmethod
    def create_pose_request(self, candidate: PoseCandidate) -> PoseRequest | None:
        """
        :param candidate: The object hypothesis to decide on.
        :return: The request to pose it, or ``None`` if it gets no pose.
        """


@dataclass
class EstimateUnposedObjects(PoseUpdateMode):
    """
    Only objects without a :class:`PoseAnnotation` are estimated.
    """

    def create_pose_request(self, candidate: PoseCandidate) -> PoseRequest | None:
        if candidate.pose_annotation is not None:
            return None
        return candidate.to_request()


@dataclass
class EstimateAllObjects(PoseUpdateMode):
    """
    Every object is estimated, ignoring the poses it already has.
    """

    def create_pose_request(self, candidate: PoseCandidate) -> PoseRequest | None:
        return candidate.to_request()
