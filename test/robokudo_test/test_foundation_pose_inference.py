"""
The FoundationPose inference on the GPU.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from pathlib import Path

import numpy
import pytest
import trimesh

pytest.importorskip(
    "robokudo_foundation_pose_tensorrt",
    reason="needs the GPU build of the ROS package robokudo_foundation_pose_tensorrt",
)

from robokudo.annotators.foundation_pose_inference import (  # noqa: E402
    FoundationPoseInference,
)
from robokudo.annotators.object_pose_estimation.object_observation import (  # noqa: E402
    ObjectObservation,
)

# %% Scene


@dataclass
class PlateOnPlane:
    """
    A thin square plate lying on the plane of the warm-up observation, covering its
    mask.
    """

    mesh_file: Path
    """
    File the plate mesh is written to.
    """

    thickness: float = 0.01
    """
    Thickness of the plate, in meters.
    """

    object_observation: ObjectObservation = field(
        default_factory=lambda: FoundationPoseInference.warm_up_object_observation
    )
    """
    The plane with the plate's mask.
    """

    def __post_init__(self) -> None:
        """
        Write the plate mesh, sized to cover the mask on the plane.
        """
        region = self.object_observation.mask_region
        size = (
            region.width
            * self.plane_depth
            / self.object_observation.frame.camera_model.focal_length_x
        )
        trimesh.creation.box(extents=(size, size, self.thickness)).export(
            self.mesh_file
        )

    @property
    def plane_depth(self) -> float:
        """
        Distance of the plane from the camera, in meters.
        """
        region = self.object_observation.mask_region
        return float(
            self.object_observation.frame.depth_image[region.pos.y, region.pos.x]
        )


# %% Fixtures


@pytest.fixture(scope="module")
def inference() -> FoundationPoseInference:
    """
    The inference with the default engines, shared by all tests of the module.
    """
    return FoundationPoseInference()


@pytest.fixture
def plate(inference: FoundationPoseInference, tmp_path: Path) -> PlateOnPlane:
    """
    A plate whose mesh the inference has loaded and selected.
    """
    plate = PlateOnPlane(mesh_file=tmp_path / "plate.obj")
    inference.load_mesh_files([plate.mesh_file])
    inference.select_mesh_file(plate.mesh_file)
    return plate


# %% Pose estimation on the GPU


def test_plate_is_posed_on_the_plane_at_its_mask(
    inference: FoundationPoseInference, plate: PlateOnPlane
) -> None:
    """
    The estimated position is the center of the plate lying on the plane.
    """
    pose = inference.estimate_pose(plate.object_observation)

    numpy.testing.assert_allclose(
        pose.position.to_np()[:3],
        [0.0, 0.0, plate.plane_depth + plate.thickness / 2],
        atol=0.01,
    )


def test_tracking_an_unmoved_plate_keeps_its_position(
    inference: FoundationPoseInference, plate: PlateOnPlane
) -> None:
    """
    Refining a correct pose on the same frame does not move it.
    """
    pose = inference.estimate_pose(plate.object_observation)

    tracked_pose = inference.track_pose(plate.object_observation, pose)

    numpy.testing.assert_allclose(
        tracked_pose.position.to_np(), pose.position.to_np(), atol=0.005
    )
