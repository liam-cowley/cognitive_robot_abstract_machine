"""
A scene and pose helpers shared by the tests of object pose estimation.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from pathlib import Path

import numpy
from krrood_spatial_math.spatial_types import Pose
from robokudo.annotators.object_pose_estimation.object_class_mesh import ObjectClassMesh
from robokudo.annotators.object_pose_estimation.object_pose_estimator import (
    ObjectPoseEstimator,
)
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification, PoseAnnotation
from robokudo.types.camera import CameraObservation
from robokudo.types.cv import ColorToDepthRatio, Point2D, Rect
from robokudo.types.scene import ObjectHypothesis
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Mesh, Scale
from semantic_digital_twin.world_description.world_entity import Body

# %% Poses


def object_pose_at_depth(depth: float) -> Pose:
    """
    :return: Pose of an object on the optical axis, ``depth`` meters in front of the camera.
    """
    return Pose.from_xyz_quaternion(pos_z=depth)


def add_pose_annotation(
    object_hypothesis: ObjectHypothesis, pose: Pose
) -> PoseAnnotation:
    """
    Give an object hypothesis a pose annotation of an unrotated pose.

    :param pose: Pose whose translation the annotation gets.
    """
    pose_annotation = PoseAnnotation()
    pose_annotation.translation = pose.position.to_np()[:3].tolist()
    pose_annotation.rotation = [0.0, 0.0, 0.0, 1.0]
    object_hypothesis.annotations.append(pose_annotation)
    return pose_annotation


def pose_annotations(object_hypothesis: ObjectHypothesis) -> list[PoseAnnotation]:
    """
    :return: The pose annotations of an object hypothesis, in order.
    """
    return [
        annotation
        for annotation in object_hypothesis.annotations
        if isinstance(annotation, PoseAnnotation)
    ]


# %% Scene


@dataclass
class FlatSurfaceScene:
    """
    A camera looking at a flat surface one meter away, with classified objects on it.
    """

    class_name: str = "box"
    """
    Class of the objects the annotator has a mesh for.
    """

    image_resolution: CameraResolution = field(
        default_factory=lambda: CameraResolution(width=64, height=48)
    )
    """
    Resolution of the color and the depth image.
    """

    object_region: Rect = field(
        default_factory=lambda: Rect(pos=Point2D(x=28, y=20), width=8, height=8)
    )
    """
    Color image region every object covers, around the image center.
    """

    def fill_cas(self, annotator: ObjectPoseEstimator) -> None:
        """
        Give the CAS of the annotator a new frame of the surface.
        """
        cas = annotator.get_cas()
        world = World()
        root = Body(name=PrefixedName(name="world"))
        with world.modify_world():
            world.add_body(root)
        camera_model = PinholeCameraModel.from_field_of_view(
            resolution=self.image_resolution,
            field_of_view=FieldOfView(
                horizontal_angle=numpy.radians(60.0),
                vertical_angle=numpy.radians(45.0),
            ),
        )
        cas.camera_observation = CameraObservation(
            camera=Camera(
                name=PrefixedName(name="camera"), root=root, camera_model=camera_model
            ),
            effective_camera_model=camera_model,
            world_T_camera=None,
            timestamp_nanoseconds=0,
        )
        shape = (self.image_resolution.height, self.image_resolution.width)
        cas.set(CASViews.COLOR_IMAGE, numpy.zeros((*shape, 3), dtype=numpy.uint8))
        cas.set(
            CASViews.DEPTH_IMAGE,
            numpy.full(
                shape,
                annotator.descriptor.parameters.depth_units_per_meter,
                dtype=numpy.uint16,
            ),
        )
        cas.color_to_depth_ratio = ColorToDepthRatio()

    def add_object_hypothesis(
        self, annotator: ObjectPoseEstimator, class_name: str | None = None
    ) -> ObjectHypothesis:
        """
        Add a classified hypothesis with a full mask over :attr:`object_region` to the
        CAS of the annotator.

        :param class_name: Class of the hypothesis, :attr:`class_name` if ``None``.
        """
        object_hypothesis = ObjectHypothesis()
        object_hypothesis.roi.roi = Rect(
            pos=Point2D(x=self.object_region.pos.x, y=self.object_region.pos.y),
            width=self.object_region.width,
            height=self.object_region.height,
        )
        object_hypothesis.roi.mask = numpy.full(
            (self.object_region.height, self.object_region.width),
            numpy.iinfo(numpy.uint8).max,
            dtype=numpy.uint8,
        )
        object_hypothesis.annotations.append(
            Classification(classname=class_name or self.class_name, confidence=1.0)
        )
        annotator.get_cas().annotations.append(object_hypothesis)
        return object_hypothesis

    def class_mesh(self, mesh_file: Path, scale: float = 1.0) -> ObjectClassMesh:
        """
        :param mesh_file: The mesh file.
        :param scale: Uniform factor bringing the file coordinates to meters.
        :return: The mesh of the objects of the scene.
        """
        return ObjectClassMesh(
            class_name=self.class_name,
            shape=Mesh(filename=str(mesh_file), scale=Scale(x=scale, y=scale, z=scale)),
        )

    def next_frame(self, annotator: ObjectPoseEstimator) -> None:
        """
        Start a new CAS in the pipeline of the annotator, holding a new frame.
        """
        annotator.get_parent_pipeline().create_new_cas()
        self.fill_cas(annotator)
