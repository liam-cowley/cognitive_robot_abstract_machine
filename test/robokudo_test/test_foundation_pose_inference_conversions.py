"""
The pose conversions of the FoundationPose inference, on engines that need no GPU.
"""

from __future__ import annotations

# %% Imports
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import numpy
import pytest
from krrood_spatial_math.spatial_types import Pose
from typing_extensions import Any, Iterator

# %% Engines without a GPU


@dataclass
class PresetPoseEngine:
    """
    Stands in for the GPU engines: answers with a preset pose matrix and records the
    arguments of every call.
    """

    refine_engine_file_path: Path
    """
    The refine engine.
    """

    score_engine_file_path: Path
    """
    The score engine.
    """

    refine_iterations: int
    """
    Refinement steps of a pose estimate.
    """

    min_depth: float
    """
    Nearest depth measurement used, in meters.
    """

    max_depth: float
    """
    Farthest depth measurement used, in meters.
    """

    pose_matrix: numpy.ndarray = field(
        default_factory=lambda: numpy.array(
            [
                [0.0, -1.0, 0.0, 0.1],
                [1.0, 0.0, 0.0, 0.2],
                [0.0, 0.0, 1.0, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    """
    Pose matrix of every answer.
    """

    track_arguments: dict[str, Any] = field(default_factory=dict)
    """
    Arguments of the last :meth:`track` call.
    """

    def estimate(self, **_: Any) -> tuple[numpy.ndarray, float, float]:
        """
        :return: The preset pose matrix with placeholders of the unused results.
        """
        return self.pose_matrix, 0.0, 0.0

    def track(self, **arguments: Any) -> tuple[numpy.ndarray, float, float]:
        """
        :return: The preset pose matrix with placeholders of the unused results.
        """
        self.track_arguments = arguments
        return self.pose_matrix, 0.0, 0.0


# %% Fixtures


@pytest.fixture
def inference_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """
    The inference module, imported against engines that need no GPU and forgotten
    afterwards, so that no other test sees those engines.
    """
    inference_module_name = "robokudo.annotators.foundation_pose_inference"
    engines = ModuleType("robokudo_foundation_pose_tensorrt")
    engines.FoundationPose = PresetPoseEngine
    engines.ensure_engines = lambda precision: (Path("refine"), Path("score"))
    monkeypatch.setitem(sys.modules, engines.__name__, engines)
    monkeypatch.delitem(sys.modules, inference_module_name, raising=False)
    yield importlib.import_module(inference_module_name)
    sys.modules.pop(inference_module_name, None)


# %% Pose conversions


def test_estimate_returns_the_pose_of_the_engine_matrix(
    inference_module: ModuleType,
) -> None:
    """
    The matrix the engine answers with becomes an equal pose.
    """
    inference = inference_module.FoundationPoseInference()

    pose = inference.estimate_pose(inference.warm_up_object_observation)

    numpy.testing.assert_allclose(pose.to_np(), inference.foundation_pose.pose_matrix)


def test_track_hands_the_engine_a_matrix_and_returns_a_pose(
    inference_module: ModuleType,
) -> None:
    """
    A track converts its pose to a matrix for the engine and back.
    """
    inference = inference_module.FoundationPoseInference()
    previous_pose = Pose.from_xyz_quaternion(pos_z=0.5)

    pose = inference.track_pose(inference.warm_up_object_observation, previous_pose)

    numpy.testing.assert_allclose(
        inference.foundation_pose.track_arguments["previous_pose"],
        previous_pose.to_np(),
    )
    numpy.testing.assert_allclose(pose.to_np(), inference.foundation_pose.pose_matrix)


def test_usable_depth_range_reaches_the_engine(inference_module: ModuleType) -> None:
    """
    The engine ignores the same depths the range declares unusable.
    """
    inference = inference_module.FoundationPoseInference()

    assert (
        inference.foundation_pose.min_depth
        == inference.usable_depth_range.minimum_distance
    )
    assert (
        inference.foundation_pose.max_depth
        == inference.usable_depth_range.maximum_distance
    )
