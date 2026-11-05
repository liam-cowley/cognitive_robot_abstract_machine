"""
Object pose estimation with single capabilities.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from pathlib import Path

import numpy
import open3d
import pytest
import trimesh
from krrood_spatial_math.spatial_types import Pose
from py_trees.common import Status
from robokudo.annotators.object_pose_estimation.exceptions import (
    PoseTrackingNotSupported,
)
from robokudo.annotators.object_pose_estimation.mesh_based_pose_estimation import (
    MeshBasedPoseEstimation,
)
from robokudo.annotators.object_pose_estimation.object_pose_estimator import (
    ObjectPoseEstimator,
)
from robokudo.annotators.object_pose_estimation.pose_tracking import (
    EstimateAndTrack,
    PoseTracking,
    TrackingPoseRequest,
)
from robokudo.annotators.object_pose_estimation.pose_update_mode import PoseRequest
from robokudo.annotators.object_pose_estimation.pose_visualization import (
    DrawableGeometryKey,
)
from robokudo.annotators.object_pose_visualizer import ObjectPoseVisualizer
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.cas import CASViews
from robokudo.pipeline import Pipeline
from robokudo.utils.error_handling import (
    clear_blackboard_exception,
    get_blackboard_exception,
)
from robokudo.utils.type_conversion import get_transform_matrix_from_pose_annotation
from typing_extensions import Iterator

from .object_pose_scene import (
    FlatSurfaceScene,
    add_pose_annotation,
    object_pose_at_depth,
    pose_annotations,
)

# %% Estimators with single capabilities


@dataclass(eq=False)
class EstimatingOnly(ObjectPoseEstimator):
    """
    Estimates a preset pose, with neither meshes nor tracking.
    """

    estimated_pose: Pose = field(
        kw_only=True, default_factory=lambda: object_pose_at_depth(0.5)
    )
    """
    Pose every estimate returns.
    """

    def estimate_object_pose(self, request: PoseRequest) -> Pose | None:
        return self.estimated_pose


@dataclass(eq=False)
class EstimatingAndTracking(PoseTracking, EstimatingOnly):
    """
    Also tracks to a preset pose, without meshes.
    """

    tracked_pose: Pose = field(
        kw_only=True, default_factory=lambda: object_pose_at_depth(0.6)
    )
    """
    Pose every track returns.
    """

    def track_object_pose(self, request: TrackingPoseRequest) -> Pose:
        return self.tracked_pose


@dataclass(eq=False)
class EstimatingWithMeshes(MeshBasedPoseEstimation, EstimatingOnly):
    """
    Also poses by meshes, without tracking.
    """

    selected_mesh_files: list[Path] = field(default_factory=list)
    """
    Mesh file of every :meth:`select_mesh_file` call, in order.
    """

    def load_mesh_files(self, mesh_files: list[Path]) -> None:
        pass

    def select_mesh_file(self, mesh_file: Path) -> None:
        self.selected_mesh_files.append(mesh_file)


def in_pipeline(
    annotator: ObjectPoseEstimator, scene: FlatSurfaceScene
) -> ObjectPoseEstimator:
    """
    :return: The annotator, in a pipeline whose CAS holds a frame of the scene.
    """
    pipeline = Pipeline("ObjectPoseEstimation")
    pipeline.add_child(annotator)
    pipeline.setup()
    scene.fill_cas(annotator)
    return annotator


# %% Fixtures


@pytest.fixture(autouse=True)
def blackboard_without_exception() -> Iterator[None]:
    """
    Keep an exception of one test from failing the annotators of the next.
    """
    yield
    clear_blackboard_exception()


# %% Estimation without meshes


def test_object_of_any_class_is_estimated_without_meshes(
    scene: FlatSurfaceScene,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    Without the mesh capability, no class is left out.
    """
    annotator = in_pipeline(EstimatingOnly(), scene)
    object_hypothesis = scene.add_object_hypothesis(annotator, class_name="any class")

    annotator.compute()

    (pose_annotation,) = pose_annotations(object_hypothesis)
    numpy.testing.assert_allclose(
        get_transform_matrix_from_pose_annotation(pose_annotation),
        annotator.estimated_pose.to_np(),
    )


def test_pose_without_mesh_is_shown_by_the_pose_visualizer(
    scene: FlatSurfaceScene,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    An estimator without meshes draws nothing itself; its annotations appear in the
    visualizer.
    """
    annotator = in_pipeline(EstimatingOnly(), scene)
    visualizer = ObjectPoseVisualizer()
    annotator.get_parent_pipeline().add_child(visualizer)
    annotator.get_cas().set(CASViews.CLOUD, open3d.geometry.PointCloud())
    scene.add_object_hypothesis(annotator)

    annotator.compute()
    visualizer.update()

    assert annotator.get_annotator_output_struct().geometries == []
    (_, pose_axes) = visualizer.get_annotator_output_struct().geometries
    numpy.testing.assert_allclose(
        pose_axes[DrawableGeometryKey.GEOMETRY].get_center()[2],
        annotator.estimated_pose.position.to_np()[2],
        atol=0.1,
    )


# %% Tracking capability


def test_tracking_mode_is_rejected_on_construction_without_the_tracking_capability() -> (
    None
):
    """
    Misconfiguration fails when the annotator is built, before any frame is processed.
    """
    descriptor = EstimatingOnly.Descriptor()
    descriptor.parameters.pose_update_mode = EstimateAndTrack()

    with pytest.raises(PoseTrackingNotSupported):
        EstimatingOnly(descriptor=descriptor)


def test_tracking_mode_set_later_fails_the_frame_without_the_tracking_capability(
    scene: FlatSurfaceScene,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    Changing to a tracking mode after construction is caught before any object is posed.
    """
    annotator = in_pipeline(EstimatingOnly(), scene)
    annotator.descriptor.parameters.pose_update_mode = EstimateAndTrack()
    object_hypothesis = scene.add_object_hypothesis(annotator)

    status = annotator.compute()

    assert status == Status.FAILURE
    assert isinstance(get_blackboard_exception(), PoseTrackingNotSupported)
    assert pose_annotations(object_hypothesis) == []


def test_known_pose_is_tracked_without_meshes(
    scene: FlatSurfaceScene,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    The tracking capability works without the mesh capability.
    """
    annotator = in_pipeline(EstimatingAndTracking(), scene)
    annotator.descriptor.parameters.pose_update_mode = EstimateAndTrack()
    object_hypothesis = scene.add_object_hypothesis(annotator)
    add_pose_annotation(object_hypothesis, object_pose_at_depth(0.4))

    annotator.compute()

    numpy.testing.assert_allclose(
        get_transform_matrix_from_pose_annotation(
            pose_annotations(object_hypothesis)[-1]
        ),
        annotator.tracked_pose.to_np(),
    )


# %% Mesh capability


def test_mesh_capability_poses_only_classes_with_a_mesh_without_tracking(
    scene: FlatSurfaceScene,
    tmp_path: Path,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    The mesh capability works without the tracking capability.
    """
    mesh_file = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.04, 0.06, 0.1)).export(mesh_file)
    descriptor = EstimatingWithMeshes.Descriptor()
    descriptor.parameters.class_meshes = [scene.class_mesh(mesh_file)]
    annotator = in_pipeline(EstimatingWithMeshes(descriptor=descriptor), scene)
    known_class = scene.add_object_hypothesis(annotator)
    unknown_class = scene.add_object_hypothesis(annotator, class_name="unknown")

    annotator.compute()

    assert annotator.selected_mesh_files == [mesh_file]
    (pose_annotation,) = pose_annotations(known_class)
    numpy.testing.assert_allclose(
        get_transform_matrix_from_pose_annotation(pose_annotation),
        annotator.estimated_pose.to_np(),
    )
    assert pose_annotations(unknown_class) == []
