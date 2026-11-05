"""
Drawing of posed meshes into a color image and a 3D view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import open3d
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Color
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.object_pose_estimation.pose_visualization import (
    GeometryKind,
    NamedGeometry,
    PoseVisualization,
)
from robokudo.utils.o3d_helper import create_oriented_bounding_box, trimesh_to_o3d_mesh

if TYPE_CHECKING:
    from krrood_spatial_math.spatial_types import Pose
    from semantic_digital_twin.world_description.geometry import Mesh


# %% Visualization frames


class VisualizationFrame(Enum):
    """
    Frame the axes and the bounding box of a posed mesh are drawn in.
    """

    MESH = auto()
    """
    The frame of the mesh, with the bounding box aligned to its axes.
    """

    SMALLEST_BOUNDING_BOX = auto()
    """
    The frame of the smallest oriented bounding box of the mesh.
    """


# %% Mesh visualization


@dataclass
class MeshPoseVisualization(PoseVisualization):
    """
    The color image and the 3D geometries showing posed meshes.
    """

    visualization_frame: VisualizationFrame = VisualizationFrame.MESH
    """
    Frame the axes and the bounding box are drawn in.
    """

    center_axes_on_bounding_box: bool = False
    """
    Whether the axes start at the center of the smallest oriented bounding box instead of
    the origin of the mesh, when drawn in :attr:`VisualizationFrame.MESH`.
    """

    show_object_mesh: bool = False
    """
    Whether the posed mesh is added to the 3D view.
    """

    object_mesh_color: Color = field(
        default_factory=lambda: Color.from_rgba(red=0.6, green=0.1, blue=0.9)
    )
    """
    Color of the posed mesh in the 3D view.
    """

    def draw_mesh(self, shape: Mesh, pose: Pose, label: str) -> None:
        """
        Draw a posed mesh into the color image and add it to the 3D view.

        :param shape: The mesh.
        :param pose: Pose of the mesh in the camera frame.
        :param label: Unique text identifying the object.
        """
        bounding_box = self.compute_bounding_box(shape, pose)
        camera_T_axes = self.compute_camera_T_axes(shape, pose)
        self.draw_bounding_box(bounding_box, label)
        self.draw_axes(camera_T_axes)
        self.geometries.append(
            NamedGeometry(f"{label} {GeometryKind.BOUNDING_BOX}", bounding_box)
        )
        self.add_axes_geometry(camera_T_axes, label)
        if self.show_object_mesh:
            self.geometries.append(
                NamedGeometry(
                    f"{label} {GeometryKind.OBJECT_MESH}",
                    self.create_posed_mesh(shape, pose),
                )
            )

    def compute_bounding_box(
        self, shape: Mesh, pose: Pose
    ) -> open3d.geometry.OrientedBoundingBox:
        """
        :param shape: The mesh.
        :param pose: Pose of the mesh in the camera frame.
        :return: The bounding box of the posed mesh in :attr:`visualization_frame`.
        """
        if self.visualization_frame is VisualizationFrame.SMALLEST_BOUNDING_BOX:
            smallest_box = shape.mesh.bounding_box_oriented.primitive
            camera_T_box = pose.homogeneous_matrix @ HomogeneousTransformationMatrix(
                data=smallest_box.transform
            )
            return create_oriented_bounding_box(
                smallest_box.extents, camera_T_box.to_np()
            )
        mesh_box = shape.local_frame_bounding_box
        camera_T_box = (
            pose.homogeneous_matrix
            @ HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=mesh_box.center
            )
        )
        return create_oriented_bounding_box(
            mesh_box.scale.to_np(), camera_T_box.to_np()
        )

    def compute_camera_T_axes(
        self, shape: Mesh, pose: Pose
    ) -> HomogeneousTransformationMatrix:
        """
        :param shape: The mesh.
        :param pose: Pose of the mesh in the camera frame.
        :return: Pose of the drawn axes in the camera frame.
        """
        mesh_T_box = HomogeneousTransformationMatrix(
            data=shape.mesh.bounding_box_oriented.primitive.transform
        )
        if self.visualization_frame is VisualizationFrame.SMALLEST_BOUNDING_BOX:
            return pose.homogeneous_matrix @ mesh_T_box
        if not self.center_axes_on_bounding_box:
            return pose.homogeneous_matrix
        return pose.homogeneous_matrix @ mesh_T_box.translation_matrix

    def create_posed_mesh(
        self, shape: Mesh, pose: Pose
    ) -> open3d.geometry.TriangleMesh:
        """
        :param shape: The mesh.
        :param pose: Pose of the mesh in the camera frame.
        :return: The mesh in the camera frame, painted in :attr:`object_mesh_color`.
        """
        posed_mesh = trimesh_to_o3d_mesh(shape.mesh)
        posed_mesh.paint_uniform_color(self.object_mesh_color.to_rgb())
        posed_mesh.transform(pose.to_np())
        return posed_mesh
