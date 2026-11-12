"""
What every demonstration of a Tiago handling a T-shirt lying on a table in MuJoCo shares:
the T-shirt, the physics it is simulated with, and holding and letting go of it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

import mujoco
import numpy
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor, PlanExecutor
from coraplex.robot_plans.actions.core.robot_body import MoveManipulatorAction
from cramph.context import ContextExtension
from experiments.t_shirt_demo.exceptions import TShirtNotPinned, TShirtNotUnpinned
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose
from physics_simulators.base_simulator import SimulatorCallbackResult
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.robots.robot_parts import AbstractRobot, EndEffector
from semantic_digital_twin.robots.tiago import TiagoMujoco
from semantic_digital_twin.semantic_annotations.semantic_annotations import TShirt
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.cloth import (
    Cotton,
    Fabric,
    GarmentFile,
)
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% the scene


class SceneEntity(StrEnum):
    """
    The names of what the scenes of the T-shirt demonstrations hold.
    """

    FLOOR = "floor"
    TABLE = "table"
    T_SHIRT = "t_shirt"


# %% the demonstration


@dataclass
class TShirtHandlingDemonstration(RobotDemonstration, ABC):
    """
    A Tiago handling a T-shirt that starts lying spread out on a table.

    A gripper holds the T-shirt by pinning the vertices between its fingertips to it, and
    lets go of it by unpinning them.
    """

    used_robot: type[AbstractRobot] = TiagoMujoco
    """
    The Tiago as MuJoCo's own model describes it, whose hand can hold something.
    """

    executor_type: type[PlanExecutor] = PhysicallySimulatedPlanExecutor
    """
    Performs the plan in physics, since the T-shirt exists only there.
    """

    t_shirt_mass: float = 0.255
    """
    The mass of the T-shirt, in kilograms, as RGBench states it.
    """

    t_shirt_garment: GarmentFile = GarmentFile.T_SHIRT
    """
    The mesh the T-shirt is made from.
    """

    t_shirt_fabric: Fabric = field(default_factory=Cotton)
    """
    The kind of cloth the T-shirt is made of.
    """

    t_shirt_color: Color = field(
        default_factory=lambda: Color(hue=0.6, saturation=0.7, value=0.8)
    )
    """
    The color of the T-shirt.
    """

    t_shirt_drop_height: float = 0.01
    """
    How far above the table top the T-shirt is laid out, to settle onto it, in metres.
    """

    approach_height: float = 0.1
    """
    How far above the T-shirt the fingertips come down from, in metres.
    """

    fingertip_clearance: float = 0.005
    """
    How far above the table top the fingertips close on the T-shirt, in metres.
    """

    closing_duration: float = 1.5
    """
    How long a gripper is given to close on the T-shirt before it holds it, in seconds
    of simulated time: a motion counts as done once it is commanded, before the servos
    got there.
    """

    pinning_radius: float = 0.03
    """
    How far from the point the fingertips meet at a vertex of the T-shirt may lie to be
    held by the gripper, in metres.
    """

    @property
    @abstractmethod
    def world_T_t_shirt(self) -> HomogeneousTransformationMatrix:
        """
        :return: Where the T-shirt starts lying: :attr:`GarmentFile.T_SHIRT` lays it out
            with its neckline at its origin, its hem along its x-axis and its sleeves
            along its y-axis.
        """

    def robot_specification(self) -> RobotSpecification:
        return RobotSpecification(self.used_robot)

    def is_scene_populated(self, world: World) -> bool:
        return bool(world.get_semantic_annotations_by_type(TShirt))

    def populate_scene(self, world: World) -> None:
        """
        Lay the T-shirt spread out where :attr:`world_T_t_shirt` places it.
        """
        t_shirt = TShirt.get_annotation_specification(
            SceneEntity.T_SHIRT,
            TShirt.get_default_root_kinematic_structure_entity_specification(),
            annotation_kwargs={"fabric": self.t_shirt_fabric},
        ).spawn(world, parent_T_self=self.world_T_t_shirt)
        with world.modify_world():
            t_shirt.add_cloth_sheet(
                surface=self.t_shirt_garment.create_surface(self.world_T_t_shirt),
                mass=self.t_shirt_mass,
                color=self.t_shirt_color,
            )

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0])]

    def build_simulation(self, world: World) -> MujocoSim:
        """
        :meth:`RobotDemonstration.build_simulation`, integrated and solved the way cloth
        simulates at an affordable cost.
        """
        return MujocoSim(
            world=world,
            headless=self.headless,
            integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            solver=mujoco.mjtSolver.mjSOL_CG,
        )

    # %% moving a gripper

    @staticmethod
    def gripper_pose_in(
        world_T_frame: HomogeneousTransformationMatrix, x: float, y: float, z: float
    ) -> Pose:
        """
        :param world_T_frame: The frame the gripper's position is given in.
        :param x: Where the gripper is to be along the frame's x-axis, in metres.
        :param y: Where the gripper is to be along the frame's y-axis, in metres.
        :param z: How high above the frame the gripper is to be, in metres.
        :return: The gripper there in the world, its tool frame's x-axis pointing down
            and its y-axis, which its fingers close along, along the frame's x-axis.
        """
        return (
            world_T_frame
            @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x, y, z, roll=numpy.pi / 2, pitch=numpy.pi / 2
            )
        ).pose

    @staticmethod
    def move_gripper_to(
        target_pose: Pose, end_effector: EndEffector
    ) -> MoveManipulatorAction:
        """
        :param target_pose: Where the gripper is to be.
        :param end_effector: The gripper moved.
        :return: The action moving the gripper there, allowed to touch the T-shirt and
            the table.
        """
        return MoveManipulatorAction(
            target_pose=target_pose,
            end_effector=end_effector,
            allow_gripper_collision=True,
        )

    @staticmethod
    def fingertip_reach(gripper: HasTwoFingers) -> float:
        """
        :param gripper: The gripper pointing down at the T-shirt.
        :return: How far ahead of the gripper's tool frame its fingertips reach along
            the direction it approaches in, which is the tool frame's x-axis.
        """
        return max(
            tip.collision.as_bounding_box_collection_in_frame(gripper.tool_frame)
            .bounding_box()
            .max_x
            for tip in gripper.fingertips.both
        )

    # %% holding the T-shirt

    def pin_t_shirt_to(
        self, tool_frame: KinematicStructureEntity, fingertip_reach: float
    ) -> None:
        """
        Pin the vertices of the T-shirt within :attr:`pinning_radius` of where the
        fingertips meet, ``fingertip_reach`` ahead of ``tool_frame``, to it.

        :raises TShirtNotPinned: If no vertex lies that close.
        """
        result = self.simulation.simulator.pin_flex_vertices(
            flex_name=SceneEntity.T_SHIRT,
            body_name=tool_frame.name.name,
            body_P_centre=numpy.array([fingertip_reach, 0.0, 0.0]),
            radius=self.pinning_radius,
        )
        if (
            result.type
            is not SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
        ):
            raise TShirtNotPinned(
                t_shirt_name=SceneEntity.T_SHIRT,
                holding_body_name=tool_frame.name.name,
                reason=result.info,
            )

    def unpin_t_shirt_from(self, tool_frame: KinematicStructureEntity) -> None:
        """
        Let go of the vertices of the T-shirt :meth:`pin_t_shirt_to` pinned to
        ``tool_frame``.

        :raises TShirtNotUnpinned: If ``tool_frame`` holds no vertex of the T-shirt.
        """
        result = self.simulation.simulator.unpin_flex_vertices(
            flex_name=SceneEntity.T_SHIRT, body_name=tool_frame.name.name
        )
        if (
            result.type
            is not SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_MODEL
        ):
            raise TShirtNotUnpinned(
                t_shirt_name=SceneEntity.T_SHIRT,
                holding_body_name=tool_frame.name.name,
                reason=result.info,
            )
