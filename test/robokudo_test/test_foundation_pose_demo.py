"""
Tests of the meshes the FoundationPose demo poses objects with.
"""

from __future__ import annotations

import pytest
from ament_index_python.packages import get_packages_with_prefixes

pytest.importorskip(
    "robokudo_foundation_pose_tensorrt",
    reason="needs the GPU build of the ROS package robokudo_foundation_pose_tensorrt",
)

from robokudo.descriptors.analysis_engines.foundation_pose_demo import (  # noqa: E402
    TexturedObject,
)


# %% Skip condition

pytestmark = pytest.mark.skipif(
    "iai_objects" not in get_packages_with_prefixes(),
    reason="needs the ROS package iai_objects",
)


# %% Textured object meshes


@pytest.mark.parametrize("textured_object", list(TexturedObject))
def test_textured_mesh_is_posed_from_its_own_file(
    textured_object: TexturedObject,
) -> None:
    """
    Each textured mesh is a single mesh in meters, so FoundationPose loads it unchanged.
    """
    assert textured_object.class_mesh.estimator_mesh_file == textured_object.mesh_file


@pytest.mark.parametrize("textured_object", list(TexturedObject))
def test_textured_mesh_is_found_by_its_detected_class_name(
    textured_object: TexturedObject,
) -> None:
    """
    A detection of the YOLO model is posed with the mesh of its object.
    """
    assert textured_object.class_mesh.class_name == textured_object.value
