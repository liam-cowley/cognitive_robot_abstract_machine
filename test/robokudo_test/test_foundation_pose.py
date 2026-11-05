"""
The FoundationPose annotator with scripted inference.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from pathlib import Path

import numpy
import pytest
import trimesh
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose
from py_trees.common import Status
from robokudo.annotators.foundation_pose import FoundationPoseAnnotator
from robokudo.annotators.object_pose_estimation.exceptions import (
    DuplicateObjectClassMesh,
    ImageResolutionMismatch,
)
from robokudo.annotators.object_pose_estimation.mesh_pose_visualization import (
    MeshPoseVisualization,
    VisualizationFrame,
)
from robokudo.annotators.object_pose_estimation.object_observation import (
    ColorAndDepthFrame,
    ObjectObservation,
)
from robokudo.annotators.object_pose_estimation.pose_tracking import (
    EstimateAndTrack,
    TrackPosedObjects,
)
from robokudo.annotators.object_pose_estimation.pose_update_mode import (
    EstimateAllObjects,
    EstimateUnposedObjects,
    PoseUpdateMode,
)
from robokudo.annotators.object_pose_estimation.pose_visualization import (
    DrawableGeometryKey,
    GeometryKind,
    PoseVisualization,
)
from robokudo.annotators.object_pose_estimation.previous_frame_poses import (
    PosedObject,
    PreviousFramePoses,
)
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.cas import CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.camera import CameraObservation
from robokudo.types.cv import ColorToDepthRatio, Point2D, Rect
from robokudo.utils.annotator_helper import transform_pose_from_camera_to_world
from robokudo.utils.transform import get_transform_matrix_from_q
from robokudo.utils.type_conversion import get_transform_matrix_from_pose_annotation
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body

from .object_pose_scene import (
    FlatSurfaceScene,
    add_pose_annotation,
    object_pose_at_depth,
    pose_annotations,
)

# %% Scripted pose estimation


@dataclass
class ScriptedPoseInference:
    """
    Answers every estimate and track with a preset pose and records what it was asked.
    """

    estimated_pose: Pose | None = field(
        default_factory=lambda: object_pose_at_depth(0.5)
    )
    """
    Pose every estimate returns.
    """

    tracked_pose: Pose = field(default_factory=lambda: object_pose_at_depth(0.6))
    """
    Pose every track returns.
    """

    loaded_mesh_files: list[Path] = field(default_factory=list)
    """
    Mesh files of every :meth:`load_mesh_files` call, in order.
    """

    selected_mesh_files: list[Path] = field(default_factory=list)
    """
    Mesh file of every :meth:`select_mesh_file` call, in order.
    """

    estimated_object_observations: list[ObjectObservation] = field(default_factory=list)
    """
    Observation of every :meth:`estimate_pose` call, in order.
    """

    previous_poses: list[Pose] = field(default_factory=list)
    """
    Previous pose of every :meth:`track_pose` call, in order.
    """

    def load_mesh_files(self, mesh_files: list[Path]) -> None:
        self.loaded_mesh_files.extend(mesh_files)

    def select_mesh_file(self, mesh_file: Path) -> None:
        self.selected_mesh_files.append(mesh_file)

    def estimate_pose(self, object_observation: ObjectObservation) -> Pose | None:
        self.estimated_object_observations.append(object_observation)
        return self.estimated_pose

    def track_pose(
        self, object_observation: ObjectObservation, previous_pose: Pose
    ) -> Pose:
        self.previous_poses.append(previous_pose)
        return self.tracked_pose


# %% Fixtures


@pytest.fixture
def box_mesh_file(tmp_path: Path) -> Path:
    """
    A box mesh in meters, centered on its origin.
    """
    mesh_file = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.04, 0.06, 0.1)).export(mesh_file)
    return mesh_file


@pytest.fixture
def pose_inference() -> ScriptedPoseInference:
    """
    A scripted estimator with its default poses.
    """
    return ScriptedPoseInference()


@pytest.fixture
def annotator(
    scene: FlatSurfaceScene,
    box_mesh_file: Path,
    pose_inference: ScriptedPoseInference,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> FoundationPoseAnnotator:
    """
    An annotator for the class of the scene, in a pipeline whose CAS holds a frame of
    it.
    """
    descriptor = FoundationPoseAnnotator.Descriptor()
    descriptor.parameters = descriptor.Parameters()
    descriptor.parameters.class_meshes = [scene.class_mesh(box_mesh_file)]
    annotator = FoundationPoseAnnotator(
        descriptor=descriptor, foundation_pose_inference=pose_inference
    )
    pipeline = Pipeline("FoundationPose")
    pipeline.add_child(annotator)
    pipeline.setup()
    scene.fill_cas(annotator)
    return annotator


def current_frame(annotator: FoundationPoseAnnotator) -> ColorAndDepthFrame:
    """
    :return: The frame the annotator would pose objects in.
    """
    return ColorAndDepthFrame.from_cas(
        annotator.get_cas(), annotator.descriptor.parameters.depth_units_per_meter
    )


# %% Object class meshes


def test_unscaled_single_mesh_is_estimated_from_its_own_file(
    scene: FlatSurfaceScene,
    box_mesh_file: Path,
) -> None:
    """
    A file that already holds one mesh in meters needs no rewriting.
    """
    class_mesh = scene.class_mesh(box_mesh_file)

    assert class_mesh.estimator_mesh_file == box_mesh_file


def test_scaled_mesh_is_rewritten_in_meters(
    scene: FlatSurfaceScene,
    box_mesh_file: Path,
) -> None:
    """
    A declared scale is baked into the file the inference loads.
    """
    scale = 0.5

    class_mesh = scene.class_mesh(box_mesh_file, scale=scale)

    rewritten_extents = trimesh.load(class_mesh.estimator_mesh_file).extents
    source_extents = trimesh.load(box_mesh_file).extents
    assert class_mesh.estimator_mesh_file != box_mesh_file
    numpy.testing.assert_allclose(rewritten_extents, source_extents * scale)


def test_mesh_file_declaring_its_unit_is_rewritten_in_meters(
    scene: FlatSurfaceScene,
    tmp_path: Path,
) -> None:
    """
    A unit declared by the file itself is converted without a configured scale.
    """
    box = trimesh.creation.box(extents=(100.0, 100.0, 100.0))
    box.units = "millimeters"
    mesh_file = tmp_path / "millimeters.glb"
    box.export(mesh_file)

    class_mesh = scene.class_mesh(mesh_file)

    expected_extents = box.extents * trimesh.units.unit_conversion(
        "millimeters", "meters"
    )
    numpy.testing.assert_allclose(class_mesh.shape.mesh.extents, expected_extents)
    numpy.testing.assert_allclose(
        trimesh.load_scene(class_mesh.estimator_mesh_file).extents, expected_extents
    )


def test_multi_part_mesh_is_rewritten_as_one_mesh(
    scene: FlatSurfaceScene,
    tmp_path: Path,
) -> None:
    """
    The inference reads a single mesh, so every part ends up in it.
    """
    parts = [
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)),
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)).apply_translation((0.3, 0, 0)),
    ]
    mesh_file = tmp_path / "parts.glb"
    trimesh.Scene(parts).export(mesh_file)

    class_mesh = scene.class_mesh(mesh_file)

    (rewritten,) = trimesh.load(class_mesh.estimator_mesh_file).geometry.values()
    assert len(rewritten.vertices) == sum(len(part.vertices) for part in parts)


def test_single_mesh_scene_is_estimated_from_its_own_file(
    scene: FlatSurfaceScene, tmp_path: Path
) -> None:
    """
    A scene wrapping one mesh without a transform counts as a single mesh.
    """
    mesh_file = tmp_path / "single.glb"
    trimesh.Scene([trimesh.creation.box(extents=(0.1, 0.1, 0.1))]).export(mesh_file)

    class_mesh = scene.class_mesh(mesh_file)

    assert class_mesh.estimator_mesh_file == mesh_file


def test_two_meshes_of_one_class_are_rejected(
    scene: FlatSurfaceScene,
    box_mesh_file: Path,
    pose_inference: ScriptedPoseInference,
) -> None:
    """
    The mesh of a class must be unambiguous.
    """
    descriptor = FoundationPoseAnnotator.Descriptor()
    descriptor.parameters = descriptor.Parameters()
    descriptor.parameters.class_meshes = [
        scene.class_mesh(box_mesh_file),
        scene.class_mesh(box_mesh_file),
    ]

    with pytest.raises(DuplicateObjectClassMesh):
        FoundationPoseAnnotator(
            descriptor=descriptor, foundation_pose_inference=pose_inference
        )


def test_inference_loads_every_class_mesh(
    annotator: FoundationPoseAnnotator, pose_inference: ScriptedPoseInference
) -> None:
    """
    Every configured mesh reaches the inference on construction.
    """
    assert (
        pose_inference.loaded_mesh_files == annotator.class_meshes.estimator_mesh_files
    )


# %% Frames


def test_frame_rejects_images_of_different_resolutions(
    annotator: FoundationPoseAnnotator,
) -> None:
    """
    The images of a frame must share one pixel grid.
    """
    frame = current_frame(annotator)

    with pytest.raises(ImageResolutionMismatch):
        ColorAndDepthFrame(
            color_image=frame.color_image[::2, ::2],
            depth_image=frame.depth_image,
            camera_model=frame.camera_model,
        )


# %% Matching across frames


def test_match_takes_the_most_overlapping_object_of_the_class(
    scene: FlatSurfaceScene,
) -> None:
    """
    The best match is taken and no longer available to later objects.
    """
    overlapping = PosedObject(
        class_name=scene.class_name,
        pose=object_pose_at_depth(0.5),
        mask_region=scene.object_region,
    )
    distant = PosedObject(
        class_name=scene.class_name,
        pose=object_pose_at_depth(0.7),
        mask_region=Rect(pos=Point2D(x=0, y=0), width=8, height=8),
    )
    previous_frame_poses = PreviousFramePoses(unmatched_objects=[distant, overlapping])

    pose = previous_frame_poses.take_match(scene.class_name, scene.object_region, 0.1)

    assert pose is overlapping.pose
    assert previous_frame_poses.unmatched_objects == [distant]


def test_match_ignores_objects_of_other_classes(scene: FlatSurfaceScene) -> None:
    """
    Overlap alone does not match objects of different classes.
    """
    other_class = PosedObject(
        class_name="other",
        pose=object_pose_at_depth(0.5),
        mask_region=scene.object_region,
    )
    previous_frame_poses = PreviousFramePoses(unmatched_objects=[other_class])

    assert (
        previous_frame_poses.take_match(scene.class_name, scene.object_region, 0.1)
        is None
    )


def test_match_requires_the_minimum_overlap(scene: FlatSurfaceScene) -> None:
    """
    An object overlapping too little is not matched and stays available.
    """
    distant = PosedObject(
        class_name=scene.class_name,
        pose=object_pose_at_depth(0.5),
        mask_region=Rect(pos=Point2D(x=0, y=0), width=8, height=8),
    )
    previous_frame_poses = PreviousFramePoses(unmatched_objects=[distant])

    assert (
        previous_frame_poses.take_match(scene.class_name, scene.object_region, 0.1)
        is None
    )
    assert previous_frame_poses.unmatched_objects == [distant]


# %% Pose update modes


@pytest.mark.parametrize(
    "mode, is_posed",
    [
        (EstimateUnposedObjects(), True),
        (TrackPosedObjects(), False),
    ],
)
def test_mode_gives_no_new_pose(
    annotator: FoundationPoseAnnotator,
    pose_inference: ScriptedPoseInference,
    mode: PoseUpdateMode,
    is_posed: bool,
    scene: FlatSurfaceScene,
) -> None:
    """
    Neither estimate nor track runs for an object the mode leaves out.
    """
    annotator.descriptor.parameters.pose_update_mode = mode
    object_hypothesis = scene.add_object_hypothesis(annotator)
    if is_posed:
        add_pose_annotation(object_hypothesis, object_pose_at_depth(0.4))
    annotations_before = list(object_hypothesis.annotations)

    annotator.compute()

    assert object_hypothesis.annotations == annotations_before
    assert pose_inference.estimated_object_observations == []
    assert pose_inference.previous_poses == []


@pytest.mark.parametrize(
    "mode",
    [
        EstimateAndTrack(),
        EstimateUnposedObjects(),
        EstimateAllObjects(),
    ],
)
def test_mode_estimates_object_without_pose(
    annotator: FoundationPoseAnnotator,
    pose_inference: ScriptedPoseInference,
    mode: PoseUpdateMode,
    scene: FlatSurfaceScene,
) -> None:
    """
    An object without a known pose is estimated, never tracked.
    """
    annotator.descriptor.parameters.pose_update_mode = mode
    object_hypothesis = scene.add_object_hypothesis(annotator)

    annotator.compute()

    (pose_annotation,) = pose_annotations(object_hypothesis)
    numpy.testing.assert_allclose(
        get_transform_matrix_from_pose_annotation(pose_annotation),
        pose_inference.estimated_pose.to_np(),
    )
    assert pose_inference.previous_poses == []


@pytest.mark.parametrize("mode", [EstimateAndTrack(), TrackPosedObjects()])
def test_mode_tracks_object_from_its_pose_annotation(
    annotator: FoundationPoseAnnotator,
    pose_inference: ScriptedPoseInference,
    mode: PoseUpdateMode,
    scene: FlatSurfaceScene,
) -> None:
    """
    Tracking starts at the annotated pose and its result is annotated.
    """
    annotator.descriptor.parameters.pose_update_mode = mode
    object_hypothesis = scene.add_object_hypothesis(annotator)
    known_pose = object_pose_at_depth(0.4)
    add_pose_annotation(object_hypothesis, known_pose)

    annotator.compute()

    (previous_pose,) = pose_inference.previous_poses
    numpy.testing.assert_allclose(previous_pose.to_np(), known_pose.to_np())
    numpy.testing.assert_allclose(
        get_transform_matrix_from_pose_annotation(
            pose_annotations(object_hypothesis)[-1]
        ),
        pose_inference.tracked_pose.to_np(),
    )


def test_estimate_all_objects_ignores_existing_pose(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_inference: ScriptedPoseInference,
) -> None:
    """
    A known pose does not turn the estimate into a track.
    """
    annotator.descriptor.parameters.pose_update_mode = EstimateAllObjects()
    object_hypothesis = scene.add_object_hypothesis(annotator)
    add_pose_annotation(object_hypothesis, object_pose_at_depth(0.4))

    annotator.compute()

    assert pose_inference.previous_poses == []
    numpy.testing.assert_allclose(
        get_transform_matrix_from_pose_annotation(
            pose_annotations(object_hypothesis)[-1]
        ),
        pose_inference.estimated_pose.to_np(),
    )


def test_compute_records_its_duration(annotator: FoundationPoseAnnotator) -> None:
    """
    The time of a frame is available for profiling.
    """
    annotator.compute()

    assert annotator._times["compute"] > 0


def test_object_of_unknown_class_is_not_posed(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_inference: ScriptedPoseInference,
) -> None:
    """
    A class without a mesh reaches neither the inference nor the annotations.
    """
    object_hypothesis = scene.add_object_hypothesis(annotator, class_name="unknown")

    annotator.compute()

    assert pose_annotations(object_hypothesis) == []
    assert pose_inference.estimated_object_observations == []


# %% Pose annotations


def test_pose_annotation_reaches_the_world_through_the_camera_pose_of_the_cas(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_inference: ScriptedPoseInference,
) -> None:
    """
    Annotations stay in the camera frame, and the CAS brings them to the world.
    """
    world_T_camera = get_transform_matrix_from_q(
        [0.0, 0.0, numpy.sin(numpy.pi / 4), numpy.cos(numpy.pi / 4)], [1.0, 2.0, 3.0]
    )
    camera_body = Body(name=PrefixedName(name="optical frame"))
    cas = annotator.get_cas()
    camera_model = cas.camera_observation.effective_camera_model
    cas.camera_observation = CameraObservation(
        camera=Camera(
            name=PrefixedName(name="camera"),
            root=camera_body,
            camera_model=camera_model,
        ),
        effective_camera_model=camera_model,
        world_T_camera=HomogeneousTransformationMatrix(
            data=world_T_camera,
            reference_frame=Body(name=PrefixedName(name="world")),
            child_frame=camera_body,
        ),
        timestamp_nanoseconds=0,
    )
    object_hypothesis = scene.add_object_hypothesis(annotator)

    annotator.compute()

    (pose_annotation,) = pose_annotations(object_hypothesis)
    numpy.testing.assert_allclose(
        get_transform_matrix_from_pose_annotation(
            transform_pose_from_camera_to_world(cas, pose_annotation)
        ),
        world_T_camera @ pose_inference.estimated_pose.to_np(),
        atol=1e-9,
    )
    assert pose_annotation.source == annotator.get_class_name()


def test_tracked_pose_overwrites_its_annotation_when_configured(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_inference: ScriptedPoseInference,
) -> None:
    """
    Updating in place adds no second annotation.
    """
    annotator.descriptor.parameters.pose_update_mode = EstimateAndTrack(
        update_existing_pose_annotations=True
    )
    object_hypothesis = scene.add_object_hypothesis(annotator)
    pose_annotation = add_pose_annotation(object_hypothesis, object_pose_at_depth(0.4))

    annotator.compute()

    assert pose_annotations(object_hypothesis) == [pose_annotation]
    numpy.testing.assert_allclose(
        get_transform_matrix_from_pose_annotation(pose_annotation),
        pose_inference.tracked_pose.to_np(),
    )


def test_object_without_estimated_pose_gets_no_annotation(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_inference: ScriptedPoseInference,
) -> None:
    """
    A failed estimate leaves the annotations untouched.
    """
    pose_inference.estimated_pose = None
    object_hypothesis = scene.add_object_hypothesis(annotator)

    annotator.compute()

    assert pose_annotations(object_hypothesis) == []


# %% Tracking


def test_object_is_tracked_from_its_latest_pose_annotation(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_inference: ScriptedPoseInference,
) -> None:
    """
    With several annotations, the last one is the starting point.
    """
    object_hypothesis = scene.add_object_hypothesis(annotator)
    add_pose_annotation(object_hypothesis, object_pose_at_depth(0.4))
    latest_pose = object_pose_at_depth(0.45)
    add_pose_annotation(object_hypothesis, latest_pose)

    annotator.compute()

    (previous_pose,) = pose_inference.previous_poses
    numpy.testing.assert_allclose(previous_pose.to_np(), latest_pose.to_np())


def test_object_is_tracked_from_the_previous_frame(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_inference: ScriptedPoseInference,
) -> None:
    """
    A matching object of the previous frame stands in for a missing annotation.
    """
    scene.add_object_hypothesis(annotator)
    annotator.compute()
    scene.next_frame(annotator)
    object_hypothesis = scene.add_object_hypothesis(annotator)

    annotator.compute()

    (previous_pose,) = pose_inference.previous_poses
    numpy.testing.assert_allclose(
        previous_pose.to_np(), pose_inference.estimated_pose.to_np()
    )
    numpy.testing.assert_allclose(
        get_transform_matrix_from_pose_annotation(
            pose_annotations(object_hypothesis)[0]
        ),
        pose_inference.tracked_pose.to_np(),
    )


def test_object_is_not_tracked_across_frames_when_disabled(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_inference: ScriptedPoseInference,
) -> None:
    """
    Without matching across frames, each frame estimates anew.
    """
    annotator.descriptor.parameters.pose_update_mode = EstimateAndTrack(
        track_across_frames=False
    )
    scene.add_object_hypothesis(annotator)
    annotator.compute()
    scene.next_frame(annotator)
    scene.add_object_hypothesis(annotator)

    annotator.compute()

    assert pose_inference.previous_poses == []
    assert len(pose_inference.estimated_object_observations) == 2


def test_track_leaving_the_mask_is_estimated_anew(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_inference: ScriptedPoseInference,
) -> None:
    """
    An implausible track is replaced by a new estimate.
    """
    pose_inference.tracked_pose = Pose.from_xyz_quaternion(pos_x=0.5, pos_z=0.5)
    object_hypothesis = scene.add_object_hypothesis(annotator)
    add_pose_annotation(object_hypothesis, object_pose_at_depth(0.4))

    annotator.compute()

    assert len(pose_inference.previous_poses) == 1
    numpy.testing.assert_allclose(
        get_transform_matrix_from_pose_annotation(
            pose_annotations(object_hypothesis)[-1]
        ),
        pose_inference.estimated_pose.to_np(),
    )


# %% Depth image mask


def test_object_mask_is_placed_at_depth_image_resolution(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_inference: ScriptedPoseInference,
) -> None:
    """
    A mask given in color pixels covers the same region of the depth image.
    """
    cas = annotator.get_cas()
    color_to_depth_ratio = ColorToDepthRatio(horizontal=0.5, vertical=0.5)
    cas.color_to_depth_ratio = color_to_depth_ratio
    depth_image = cas.get(CASViews.DEPTH_IMAGE)
    cas.set(CASViews.DEPTH_IMAGE, depth_image[::2, ::2].copy())
    scene.add_object_hypothesis(annotator)

    annotator.compute()

    (object_observation,) = pose_inference.estimated_object_observations
    expected_region = color_to_depth_ratio.depth_region(scene.object_region)
    assert object_observation.mask.shape == cas.get(CASViews.DEPTH_IMAGE).shape
    assert object_observation.mask_region == expected_region
    assert (
        object_observation.mask.sum() == expected_region.width * expected_region.height
    )


# %% Visualization


def visualization_of(
    annotator: FoundationPoseAnnotator, frame: VisualizationFrame
) -> MeshPoseVisualization:
    """
    :return: An empty visualization of the current frame of the annotator.
    """
    cas = annotator.get_cas()
    return MeshPoseVisualization(
        color_image=cas.get_copy(CASViews.COLOR_IMAGE),
        color_to_depth_ratio=cas.color_to_depth_ratio,
        camera_model=current_frame(annotator).camera_model,
        visualization_frame=frame,
    )


def test_mesh_frame_bounding_box_spans_the_mesh(
    scene: FlatSurfaceScene, annotator: FoundationPoseAnnotator
) -> None:
    """
    The box in the mesh frame has the extents of the mesh.
    """
    shape = annotator.class_meshes.meshes_by_class_name[scene.class_name].shape
    visualization = visualization_of(annotator, VisualizationFrame.MESH)

    bounding_box = visualization.compute_bounding_box(shape, object_pose_at_depth(0.5))

    numpy.testing.assert_allclose(bounding_box.extent, shape.mesh.extents)


def test_smallest_bounding_box_frame_uses_the_oriented_box(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
) -> None:
    """
    The box in its own frame has the extents of the oriented bounding box.
    """
    shape = annotator.class_meshes.meshes_by_class_name[scene.class_name].shape
    visualization = visualization_of(
        annotator, VisualizationFrame.SMALLEST_BOUNDING_BOX
    )

    bounding_box = visualization.compute_bounding_box(shape, object_pose_at_depth(0.5))

    numpy.testing.assert_allclose(
        bounding_box.extent, shape.mesh.bounding_box_oriented.primitive.extents
    )


def test_nothing_is_drawn_for_an_object_behind_the_camera(
    scene: FlatSurfaceScene, annotator: FoundationPoseAnnotator
) -> None:
    """
    Points behind the camera have no valid projection, so nothing is drawn.
    """
    shape = annotator.class_meshes.meshes_by_class_name[scene.class_name].shape
    visualization = visualization_of(annotator, VisualizationFrame.MESH)

    visualization.draw_mesh(shape, object_pose_at_depth(-0.5), "behind the camera")

    assert not visualization.color_image.any()


def test_axes_are_drawn_in_color_image_pixels_from_the_depth_image_calibration(
    annotator: FoundationPoseAnnotator,
) -> None:
    """
    Axes land on the color image even though the calibration is at depth resolution.
    """
    color_camera_model = current_frame(annotator).camera_model
    color_to_depth_ratio = ColorToDepthRatio(horizontal=0.5, vertical=0.5)
    visualization = PoseVisualization(
        color_image=annotator.get_cas().get_copy(CASViews.COLOR_IMAGE),
        color_to_depth_ratio=color_to_depth_ratio,
        camera_model=color_camera_model.scaled(
            scale_x=color_to_depth_ratio.horizontal,
            scale_y=color_to_depth_ratio.vertical,
        ),
    )
    depth = 0.5

    visualization.draw_axes(object_pose_at_depth(depth).homogeneous_matrix)

    intrinsic_matrix = color_camera_model.intrinsic_matrix
    center_x, center_y = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    half_axis_x = intrinsic_matrix[0, 0] * visualization.axis_length / depth / 2
    half_axis_y = intrinsic_matrix[1, 1] * visualization.axis_length / depth / 2
    maximum = Color.opencv_maximum_channel_value
    assert visualization.color_image[
        int(center_y), int(center_x + half_axis_x)
    ].tolist() == [0, 0, maximum]
    assert visualization.color_image[
        int(center_y + half_axis_y), int(center_x)
    ].tolist() == [0, maximum, 0]


def test_axes_reaching_behind_the_camera_are_not_drawn(
    annotator: FoundationPoseAnnotator,
) -> None:
    """
    A single axis end behind the camera suppresses all axes.
    """
    visualization = visualization_of(annotator, VisualizationFrame.MESH)
    pose_facing_the_camera = Pose.from_xyz_quaternion(
        0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0
    )

    visualization.draw_axes(pose_facing_the_camera.homogeneous_matrix)

    assert not visualization.color_image.any()


def test_points_are_projected_to_color_image_pixels(
    annotator: FoundationPoseAnnotator,
) -> None:
    """
    A projected point is scaled from depth to color image pixels on each axis.
    """
    camera_model = current_frame(annotator).camera_model
    color_to_depth_ratio = ColorToDepthRatio(horizontal=0.5, vertical=0.5)
    intrinsic_matrix = camera_model.intrinsic_matrix
    depth = 2.0
    offset_x = 0.25

    (pixel,) = color_to_depth_ratio.project_to_color_pixels(
        numpy.array([[offset_x, 0.0, depth]]), camera_model
    )

    expected_depth_pixel = (
        intrinsic_matrix[0, 0] * offset_x / depth + intrinsic_matrix[0, 2],
        intrinsic_matrix[1, 2],
    )
    assert pixel.tolist() == [
        int(expected_depth_pixel[0] / color_to_depth_ratio.horizontal),
        int(expected_depth_pixel[1] / color_to_depth_ratio.vertical),
    ]


def test_posed_objects_are_drawn_and_named_in_the_3d_view(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
) -> None:
    """
    Each geometry of a posed object has a name unique to the object and its kind.
    """
    annotator.descriptor.parameters.visualize_object_mesh = True
    object_hypothesis = scene.add_object_hypothesis(annotator)

    status = annotator.compute()

    output = annotator.get_annotator_output_struct()
    names = [geometry[DrawableGeometryKey.NAME] for geometry in output.geometries]
    label = f"{object_hypothesis.id}: {scene.class_name}"
    assert status == Status.SUCCESS
    assert output.image.any()
    assert names == [
        f"{label} {GeometryKind.BOUNDING_BOX}",
        f"{label} {GeometryKind.AXES}",
        f"{label} {GeometryKind.OBJECT_MESH}",
    ]
