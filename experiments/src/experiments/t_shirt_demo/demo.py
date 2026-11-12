"""
A Tiago picks a T-shirt up off a table by its shoulder and swings it from side to side,
simulated in MuJoCo, to show how the cloth behaves.

Run it as a script::
    python3 -m experiments.t_shirt_demo.demo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy
from coraplex.datastructures.trajectory import PoseTrajectory
from coraplex.plans.context_extensions import RobotAccess
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import (
    FollowToolCenterPointPathAction,
    MoveFingersToWidthAction,
    ParkArmsAction,
    SetGripperAction,
)
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.monitors import CountSimulationTimeSeconds
from cramph.statechart import Statechart
from cramph.threaded_nodes import FunctionCall
from experiments.t_shirt_demo.t_shirt_handling import (
    SceneEntity,
    TShirtHandlingDemonstration,
)
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.semantic_annotations.semantic_annotations import TShirt
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

# %% the demonstration


@dataclass
class TShirtSwingDemonstration(TShirtHandlingDemonstration):
    """
    The Tiago takes the T-shirt lying spread out on the table in front of it by the
    shoulder, turns away from the table to pull it off, and swings it from side to side.
    """

    ros_node_name: str = "t_shirt_swing_demo"
    """
    Name of the node a run registers.
    """

    table_distance: float = 0.6
    """
    How far in front of the robot's base the near edge of the table stands, in metres.
    """

    table_scale: Scale = field(default_factory=lambda: Scale(0.9, 1.4, 0.72))
    """
    The depth, width and height of the table.
    """

    t_shirt_P_left_shoulder: Point3 = field(
        default_factory=lambda: Point3(0.024, 0.176, 0.015)
    )
    """
    The middle of the seam of the T-shirt's left shoulder, as
    :attr:`GarmentFile.T_SHIRT` lays it out.
    """

    t_shirt_margin: float = 0.05
    """
    How far behind the near edge of the table the T-shirt's neckline lies, in metres.
    """

    turning_angle: float = numpy.pi / 2
    """
    How far the robot turns to its left, away from the table, with the T-shirt held, in
    radians.
    """

    hanging_distance: float = 0.4
    """
    How far in front of the robot's base the gripper holds the T-shirt while swinging
    it, in metres.
    """

    hanging_height: float = 1.0
    """
    How high the gripper holds the T-shirt while swinging it, in metres.
    """

    swing_amplitude: float = 0.2
    """
    How far to each side of the shoulder's side of the robot the gripper swings, in
    metres.
    """

    swing_count: int = 3
    """
    How often the gripper swings to the left and back to the right.
    """

    watching_duration: float = 3.0
    """
    How long the T-shirt hangs after the last swing, in seconds of simulated time.
    """

    def scene_specification(self) -> WorldSpecification:
        return WorldSpecification(
            objects=[
                BodySpecification.box(
                    SceneEntity.FLOOR,
                    Scale(4.0, 4.0, 0.02),
                    parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=-0.01),
                ),
                BodySpecification.box(
                    SceneEntity.TABLE,
                    self.table_scale,
                    parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=self.table_distance + self.table_scale.x / 2,
                        z=self.table_scale.z / 2,
                    ),
                ),
            ]
        )

    @property
    def world_T_t_shirt(self) -> HomogeneousTransformationMatrix:
        """
        On the table, its neckline :attr:`t_shirt_margin` behind the near edge.
        """
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.table_distance + self.t_shirt_margin,
            z=self.table_scale.z + self.t_shirt_drop_height,
        )

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Park the arms, take the T-shirt by the shoulder on the robot's left with the arm
        closest to it, turn away from the table, swing the T-shirt and let it hang.
        """
        world = context.world
        robot = context.require_extension(RobotAccess).robot
        world_T_table_side = HomogeneousTransformationMatrix(reference_frame=world.root)
        shoulder_pose = self._shoulder_on_the_left(world)
        arm = min(
            robot.all_arms,
            key=lambda arm: float(
                arm.end_effector.tool_frame.global_pose.position.euclidean_distance(
                    shoulder_pose.position
                )
            ),
        )
        gripper = arm.end_effector
        fingertip_reach = self.fingertip_reach(gripper)
        shoulder_x, shoulder_y = float(shoulder_pose.x), float(shoulder_pose.y)
        above_shoulder = self.gripper_pose_in(
            world_T_table_side,
            shoulder_x,
            shoulder_y,
            self.table_scale.z + self.approach_height + fingertip_reach,
        )
        world_T_turned_base = (
            robot.root.global_pose.homogeneous_matrix
            @ HomogeneousTransformationMatrix.from_xyz_rpy(yaw=self.turning_angle)
        )
        base_y_of_shoulder = float(world.transform(shoulder_pose, robot.root).y)
        swing = [
            self.gripper_pose_in(
                world_T_turned_base,
                self.hanging_distance,
                base_y_of_shoulder + side,
                self.hanging_height,
            )
            for side in [
                *[self.swing_amplitude, -self.swing_amplitude] * self.swing_count,
                0.0,
            ]
        ]
        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    ParkArmsAction(arms=robot.all_arms),
                    SetGripperAction(
                        configuration=gripper.default_configuration(GripperState.OPEN)
                    ),
                    self.move_gripper_to(above_shoulder, gripper),
                    self.move_gripper_to(
                        self.gripper_pose_in(
                            world_T_table_side,
                            shoulder_x,
                            shoulder_y,
                            self.table_scale.z
                            + self.fingertip_clearance
                            + fingertip_reach,
                        ),
                        gripper,
                    ),
                    MoveFingersToWidthAction(
                        gripper=gripper, width=self.t_shirt_fabric.thickness
                    ),
                    CountSimulationTimeSeconds(seconds=self.closing_duration),
                    FunctionCall(
                        function=partial(
                            self.pin_t_shirt_to, gripper.tool_frame, fingertip_reach
                        )
                    ),
                    self.move_gripper_to(above_shoulder, gripper),
                    self.move_gripper_to(
                        self.gripper_pose_in(
                            robot.root.global_pose.homogeneous_matrix,
                            self.hanging_distance,
                            base_y_of_shoulder,
                            self.hanging_height,
                        ),
                        gripper,
                    ),
                    NavigateAction(target_location=world_T_turned_base.pose),
                    FollowToolCenterPointPathAction(
                        target_locations=PoseTrajectory(poses=swing), arm=arm
                    ),
                    CountSimulationTimeSeconds(seconds=self.watching_duration),
                ]
            )
        )
        return statechart

    def _shoulder_on_the_left(self, world: World) -> Pose:
        """
        :param world: The world holding the T-shirt.
        :return: Where the vertex of the T-shirt closest to the middle of its shoulder on
            the robot's left lies at the start.
        """
        world_P_shoulder = self.world_T_t_shirt @ self.t_shirt_P_left_shoulder
        [t_shirt] = world.get_semantic_annotations_by_type(TShirt)
        sheet = t_shirt.cloth_sheet
        _, [closest] = sheet.starting_surface.nearest.vertex(
            [world_P_shoulder.to_np()[:3]]
        )
        return Pose.from_xyz_rpy(
            *sheet.starting_surface.vertices[closest], reference_frame=world.root
        )


# %% running it


def main(headless: bool = False) -> World:
    """
    Run the demonstration.

    :param headless: Whether the physics runs without a viewer.
    :return: The world the T-shirt was swung in.
    """
    return TShirtSwingDemonstration(headless=headless).run()


if __name__ == "__main__":
    main()
