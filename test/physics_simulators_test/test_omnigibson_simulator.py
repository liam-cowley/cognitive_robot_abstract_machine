"""
Tests of :class:`physics_simulators.omnigibson_simulator.OmniGibsonSimulator`.

They need Isaac Sim and OmniGibson, so they are skipped where those are not installed.
"""

from pathlib import Path

import numpy
import pytest
import yaml

pytest.importorskip("omnigibson")

from physics_simulators.base_simulator import SimulatorCallbackResult, SimulatorState
from physics_simulators.omnigibson_simulator import OmniGibsonSimulator

# %% fixtures


@pytest.fixture(scope="module")
def configuration_file() -> Path:
    return Path(__file__).parent / "resources" / "omnigibson_single_cube.yaml"


@pytest.fixture(scope="module")
def cube_configuration(configuration_file: Path) -> dict:
    return yaml.safe_load(configuration_file.read_text())["objects"][0]


@pytest.fixture(scope="module")
def simulator(configuration_file: Path) -> OmniGibsonSimulator:
    """
    One simulator for the whole module, because Isaac Sim launches once per process.
    """
    simulator = OmniGibsonSimulator(
        _headless=True, _step_size=1.0 / 60.0, configuration_file=configuration_file
    )
    simulator.start(simulate_in_thread=False)
    yield simulator
    simulator.stop()


@pytest.fixture
def reset_simulator(simulator: OmniGibsonSimulator) -> OmniGibsonSimulator:
    simulator.reset()
    return simulator


# %% bodies


def test_configured_object_is_a_body(
    reset_simulator: OmniGibsonSimulator, cube_configuration: dict
):
    result = reset_simulator.callbacks["get_all_body_names"]()
    assert cube_configuration["name"] in result.result


def test_body_position_is_the_configured_position(
    reset_simulator: OmniGibsonSimulator, cube_configuration: dict
):
    result = reset_simulator.callbacks["get_body_position"](cube_configuration["name"])
    assert numpy.allclose(result.result, cube_configuration["position"], atol=1e-5)


def test_body_quaternion_is_reported_as_wxyz(
    reset_simulator: OmniGibsonSimulator, cube_configuration: dict
):
    configured_xyzw = numpy.array(cube_configuration["orientation"])
    result = reset_simulator.callbacks["get_body_quaternion"](
        cube_configuration["name"]
    )
    assert numpy.allclose(result.result, numpy.roll(configured_xyzw, 1), atol=1e-5)


def test_set_body_position_round_trips(
    reset_simulator: OmniGibsonSimulator, cube_configuration: dict
):
    target = numpy.array([1.0, 2.0, 0.5])
    reset_simulator.callbacks["set_body_position"](cube_configuration["name"], target)
    result = reset_simulator.callbacks["get_body_position"](cube_configuration["name"])
    assert numpy.allclose(result.result, target, atol=1e-5)


def test_set_body_quaternion_round_trips(
    reset_simulator: OmniGibsonSimulator, cube_configuration: dict
):
    half_turn_about_z_wxyz = numpy.array([0.0, 0.0, 0.0, 1.0])
    reset_simulator.callbacks["set_body_quaternion"](
        cube_configuration["name"], half_turn_about_z_wxyz
    )
    result = reset_simulator.callbacks["get_body_quaternion"](
        cube_configuration["name"]
    )
    assert numpy.allclose(numpy.abs(result.result), half_turn_about_z_wxyz, atol=1e-5)


def test_unknown_body_fails_without_execution(reset_simulator: OmniGibsonSimulator):
    result = reset_simulator.callbacks["get_body_position"]("no_such_body")
    assert result.type is SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION


def test_primitive_object_has_no_joints(
    reset_simulator: OmniGibsonSimulator, cube_configuration: dict
):
    result = reset_simulator.callbacks["get_all_joint_names"](
        cube_configuration["name"]
    )
    assert result.result == []


# %% stepping


def test_step_advances_simulation_time_by_step_size(
    reset_simulator: OmniGibsonSimulator,
):
    assert reset_simulator.state is SimulatorState.RUNNING
    time_before_step = reset_simulator.current_simulation_time
    reset_simulator.step()
    assert reset_simulator.current_simulation_time == pytest.approx(
        time_before_step + reset_simulator.step_size
    )


def test_free_body_falls_when_stepped(
    reset_simulator: OmniGibsonSimulator, cube_configuration: dict
):
    for _ in range(30):
        reset_simulator.step()
    result = reset_simulator.callbacks["get_body_position"](cube_configuration["name"])
    assert result.result[2] < cube_configuration["position"][2]
