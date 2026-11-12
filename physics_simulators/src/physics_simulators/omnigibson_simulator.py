"""
A :class:`~physics_simulators.base_simulator.BaseSimulator` running BEHAVIOR-1K scenes
in OmniGibson on top of NVIDIA Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy
import omnigibson
import torch
from omnigibson.macros import gm
from omnigibson.objects.usd_object import USDObject
from omnigibson.prims.joint_prim import JointPrim

from physics_simulators.base_simulator import (
    BaseSimulator,
    SimulatorCallbackResult,
    SimulatorRenderer,
    SimulatorState,
)


# %% environment configuration


class EnvironmentConfigurationKey(StrEnum):
    """
    Keys of an OmniGibson environment configuration that the simulator overrides.
    """

    ENVIRONMENT = "env"
    """
    Section holding the environment's simulation rates.
    """

    ACTION_FREQUENCY = "action_frequency"
    """
    Rate, in Hz, at which one OmniGibson simulation step advances time.
    """

    RENDERING_FREQUENCY = "rendering_frequency"
    """
    Rate, in Hz, at which OmniGibson renders a frame.
    """

    PHYSICS_FREQUENCY = "physics_frequency"
    """
    Rate, in Hz, at which PhysX integrates.
    """


class ObjectRegistryKey(StrEnum):
    """
    Attributes an OmniGibson scene's object registry looks objects up by.
    """

    NAME = "name"
    """
    The unique name of an object within its scene.
    """


# %% renderer


@dataclass(init=False)
class OmniGibsonRenderer(SimulatorRenderer):
    """
    The Isaac Sim application window showing an OmniGibson scene.

    Closing the renderer leaves the Isaac Sim application running, because Isaac Sim
    cannot be launched a second time within one process.
    """

    def is_running(self) -> bool:
        return super().is_running() and omnigibson.app.is_running()


# %% simulator


@dataclass(unsafe_hash=True)
class OmniGibsonSimulator(BaseSimulator):
    """
    Runs one OmniGibson environment, such as a BEHAVIOR-1K scene with its objects.

    Each standalone object in the scene is a body, posed by its root link. One
    :meth:`step` advances the simulation by :attr:`step_size`.

    .. note:: Isaac Sim runs once per process, so only one instance can exist at a time.
    """

    configuration_file: Path = field(kw_only=True)
    """
    OmniGibson environment configuration in YAML, describing the scene, robots and
    objects to load.
    """

    _environment: omnigibson.Environment = field(init=False, repr=False, compare=False)
    """
    The loaded OmniGibson environment.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        gm.HEADLESS = self.headless
        frequency = 1.0 / self.step_size
        simulation_rates = {
            EnvironmentConfigurationKey.ENVIRONMENT: {
                EnvironmentConfigurationKey.ACTION_FREQUENCY: frequency,
                EnvironmentConfigurationKey.RENDERING_FREQUENCY: frequency,
                EnvironmentConfigurationKey.PHYSICS_FREQUENCY: frequency,
            }
        }
        self._environment = omnigibson.Environment(
            configs=[str(self.configuration_file), simulation_rates]
        )

    def start_callback(self) -> None:
        omnigibson.sim.play()
        self._renderer = SimulatorRenderer() if self.headless else OmniGibsonRenderer()

    def step_callback(self) -> None:
        if self.state != SimulatorState.RUNNING:
            return
        omnigibson.sim.step()
        self._current_number_of_steps += 1

    def reset_callback(self) -> None:
        self._environment.reset(get_obs=False)

    @property
    def current_simulation_time(self) -> float:
        return omnigibson.sim.current_time

    @staticmethod
    def to_pose_tensor(values: numpy.ndarray) -> torch.Tensor:
        """
        Convert a position or quaternion to the float tensor OmniGibson poses take.
        """
        return torch.as_tensor(numpy.asarray(values), dtype=torch.float32)

    # %% bodies

    @BaseSimulator.simulator_callback
    def get_all_body_names(self) -> SimulatorCallbackResult:
        """
        Get the names of all standalone objects in the scene.

        :return: A SimulatorCallbackResult with a list of all body names as the result
        """
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info="Getting all body names",
            result=[
                scene_object.name for scene_object in self._environment.scene.objects
            ],
        )

    @BaseSimulator.simulator_callback
    def get_body(self, body_name: str) -> SimulatorCallbackResult:
        """
        Get a standalone object of the scene by its name.

        :param body_name: The name of the object
        :return: A SimulatorCallbackResult with the OmniGibson object as the result
        """
        body = self._environment.scene.object_registry(
            ObjectRegistryKey.NAME, body_name
        )
        if body is None:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Body {body_name} not found",
            )
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting body {body_name}",
            result=body,
        )

    @BaseSimulator.simulator_callback
    def get_body_position(self, body_name: str) -> SimulatorCallbackResult:
        """
        Get the world position of a body.

        :param body_name: The name of the body
        :return: A SimulatorCallbackResult with a numpy array of shape (3,) as the result
        """
        get_body = self.get_body(body_name)
        if (
            get_body.type
            != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        ):
            return get_body
        position, _ = get_body.result.get_position_orientation()
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting body position of {body_name}",
            result=position.cpu().numpy(),
        )

    @BaseSimulator.simulator_callback
    def get_body_quaternion(self, body_name: str) -> SimulatorCallbackResult:
        """
        Get the world orientation of a body.

        :param body_name: The name of the body
        :return: A SimulatorCallbackResult with a numpy array of shape (4,) representing
            the quaternion in WXYZ format as the result
        """
        get_body = self.get_body(body_name)
        if (
            get_body.type
            != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        ):
            return get_body
        _, quaternion_xyzw = get_body.result.get_position_orientation()
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting body quaternion (WXYZ) of {body_name}",
            result=numpy.roll(quaternion_xyzw.cpu().numpy(), 1),
        )

    @BaseSimulator.simulator_callback
    def set_body_position(
        self, body_name: str, position: numpy.ndarray
    ) -> SimulatorCallbackResult:
        """
        Teleport a body to a world position, keeping its orientation.

        :param body_name: The name of the body
        :param position: The new position, shape (3,)
        :return: A SimulatorCallbackResult indicating the success or failure of the
            operation
        """
        get_body = self.get_body(body_name)
        if (
            get_body.type
            != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        ):
            return get_body
        get_body.result.set_position_orientation(position=self.to_pose_tensor(position))
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA,
            info=f"Set body {body_name} to position {position}",
        )

    @BaseSimulator.simulator_callback
    def set_body_quaternion(
        self, body_name: str, quaternion: numpy.ndarray
    ) -> SimulatorCallbackResult:
        """
        Turn a body to a world orientation, keeping its position.

        :param body_name: The name of the body
        :param quaternion: The new orientation in WXYZ format, shape (4,)
        :return: A SimulatorCallbackResult indicating the success or failure of the
            operation
        """
        get_body = self.get_body(body_name)
        if (
            get_body.type
            != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        ):
            return get_body
        get_body.result.set_position_orientation(
            orientation=self.to_pose_tensor(numpy.roll(quaternion, -1))
        )
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA,
            info=f"Set body {body_name} to quaternion (WXYZ) {quaternion}",
        )

    # %% joints

    @BaseSimulator.simulator_callback
    def get_all_joint_names(self, body_name: str) -> SimulatorCallbackResult:
        """
        Get the names of the revolute and prismatic joints of a body.

        :param body_name: The name of the body
        :return: A SimulatorCallbackResult with a list of joint names as the result
        """
        get_body = self.get_body(body_name)
        if (
            get_body.type
            != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        ):
            return get_body
        body: USDObject = get_body.result
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting all joint names of {body_name}",
            result=[
                joint_name
                for joint_name, joint in body.joints.items()
                if joint.is_single_dof
            ],
        )

    @BaseSimulator.simulator_callback
    def get_joint(self, body_name: str, joint_name: str) -> SimulatorCallbackResult:
        """
        Get a revolute or prismatic joint of a body.

        :param body_name: The name of the body
        :param joint_name: The name of the joint within the body
        :return: A SimulatorCallbackResult with the OmniGibson joint as the result
        """
        get_body = self.get_body(body_name)
        if (
            get_body.type
            != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        ):
            return get_body
        joint: JointPrim | None = get_body.result.joints.get(joint_name)
        if joint is None or not joint.is_single_dof:
            return SimulatorCallbackResult(
                type=SimulatorCallbackResult.ResultType.FAILURE_WITHOUT_EXECUTION,
                info=f"Body {body_name} has no revolute or prismatic joint {joint_name}",
            )
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting joint {joint_name} of {body_name}",
            result=joint,
        )

    @BaseSimulator.simulator_callback
    def get_joint_value(
        self, body_name: str, joint_name: str
    ) -> SimulatorCallbackResult:
        """
        Get the position of a revolute or prismatic joint of a body.

        :param body_name: The name of the body
        :param joint_name: The name of the joint within the body
        :return: A SimulatorCallbackResult with the joint position as the result
        """
        get_joint = self.get_joint(body_name, joint_name)
        if (
            get_joint.type
            != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        ):
            return get_joint
        positions, _, _ = get_joint.result.get_state()
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
            info=f"Getting joint value of {joint_name} of {body_name}",
            result=positions[0].item(),
        )

    @BaseSimulator.simulator_callback
    def set_joint_value(
        self, body_name: str, joint_name: str, value: float
    ) -> SimulatorCallbackResult:
        """
        Set the position of a revolute or prismatic joint of a body instantly.

        :param body_name: The name of the body
        :param joint_name: The name of the joint within the body
        :param value: The new joint position
        :return: A SimulatorCallbackResult indicating the success or failure of the
            operation
        """
        get_joint = self.get_joint(body_name, joint_name)
        if (
            get_joint.type
            != SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
        ):
            return get_joint
        get_joint.result.set_pos(value)
        return SimulatorCallbackResult(
            type=SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA,
            info=f"Set joint {joint_name} of {body_name} to value {value}",
        )
