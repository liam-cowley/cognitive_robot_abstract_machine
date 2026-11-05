"""
The capability of an object pose estimator to pose objects by the mesh of their class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

from robokudo.annotators.object_pose_estimation.mesh_pose_visualization import (
    MeshPoseVisualization,
    VisualizationFrame,
)
from robokudo.annotators.object_pose_estimation.object_class_mesh import (
    ObjectClassMesh,
    ObjectClassMeshes,
)
from robokudo.annotators.object_pose_estimation.object_pose_estimator import (
    ObjectPoseEstimator,
)
from robokudo.annotators.object_pose_estimation.pose_visualization import NamedGeometry
from robokudo.cas import CASViews

if TYPE_CHECKING:
    from pathlib import Path

    import numpy
    import numpy.typing as numpy_typing
    from krrood_spatial_math.spatial_types import Pose

    from robokudo.annotators.object_pose_estimation.object_observation import (
        ColorAndDepthFrame,
    )
    from robokudo.annotators.object_pose_estimation.pose_update_mode import (
        EstimatedPose,
        PoseRequest,
    )


# %% Mesh based pose estimation


@dataclass(eq=False)
class MeshBasedPoseEstimation(ObjectPoseEstimator, ABC):
    """
    An object pose estimator that poses only objects of classes with a known mesh, and
    whose poses are those of the mesh.
    """

    class Descriptor(ObjectPoseEstimator.Descriptor):
        """
        Parameters of a mesh based object pose estimator.
        """

        @dataclass
        class Parameters(ObjectPoseEstimator.Descriptor.Parameters):
            """
            Pose estimation, mesh and mesh visualization settings.
            """

            class_meshes: list[ObjectClassMesh] = field(default_factory=list)
            """
            Meshes of the object classes to pose, one per class.
            """

            visualization_frame: VisualizationFrame = VisualizationFrame.MESH
            """
            Frame the axes and the bounding box of a posed object are drawn in.
            """

            center_axes_on_bounding_box: bool = False
            """
            Whether the drawn axes start at the center of the smallest oriented bounding
            box instead of the origin of the mesh.
            """

            visualize_object_mesh: bool = False
            """
            Whether the posed mesh is shown in the 3D view.
            """

        parameters = Parameters()
        """
        Pose estimation and mesh defaults.
        """

    descriptor: Descriptor = field(default_factory=Descriptor)
    """
    Pose estimation and mesh configuration.
    """

    class_meshes: ObjectClassMeshes = field(init=False, repr=False)
    """
    Meshes of the object classes to pose.
    """

    def __post_init__(self) -> None:
        """
        Initialize the base annotator and load the meshes.

        :raises DuplicateObjectClassMesh: If a class has more than one mesh.
        """
        super().__post_init__()
        self.class_meshes = ObjectClassMeshes.from_class_meshes(
            self.descriptor.parameters.class_meshes
        )
        self.load_mesh_files(self.class_meshes.estimator_mesh_files)

    @abstractmethod
    def load_mesh_files(self, mesh_files: list[Path]) -> None:
        """
        Load meshes so that selecting any of them is immediate.

        :param mesh_files: Mesh files in meters, each holding a single mesh.
        """

    @abstractmethod
    def select_mesh_file(self, mesh_file: Path) -> None:
        """
        Use a mesh for the following pose estimates and tracks.

        :param mesh_file: Mesh file in meters, holding a single mesh.
        """

    def find_class_mesh(self, request: PoseRequest) -> ObjectClassMesh:
        """
        :param request: An object hypothesis to pose.
        :return: The mesh of its class.
        """
        return self.class_meshes.meshes_by_class_name[request.class_name]

    def can_pose_class(self, class_name: str) -> bool:
        """
        Like :meth:`ObjectPoseEstimator.can_pose_class`, but only classes with a mesh.
        """
        return class_name in self.class_meshes.meshes_by_class_name

    def compute_object_center(
        self, request: PoseRequest, pose: Pose
    ) -> numpy_typing.NDArray[numpy.float64]:
        """
        Like :meth:`ObjectPoseEstimator.compute_object_center`, but the center of the
        bounding box of the mesh.
        """
        mesh_P_center = self.find_class_mesh(
            request
        ).shape.local_frame_bounding_box.center
        return (pose.homogeneous_matrix @ mesh_P_center).to_np()[:3]

    def determine_object_pose(self, request: PoseRequest) -> Pose | None:
        """
        Like :meth:`ObjectPoseEstimator.determine_object_pose`, using the mesh of the
        class of the request.
        """
        self.select_mesh_file(self.find_class_mesh(request).estimator_mesh_file)
        return super().determine_object_pose(request)

    def visualize(
        self, frame: ColorAndDepthFrame, estimated_poses: list[EstimatedPose]
    ) -> None:
        """
        Like :meth:`ObjectPoseEstimator.visualize`, but show the posed meshes in the
        color image and, with the point cloud of the CAS, in the 3D view.
        """
        cas = self.get_cas()
        parameters = self.descriptor.parameters
        visualization = MeshPoseVisualization(
            color_image=cas.get_copy(CASViews.COLOR_IMAGE),
            color_to_depth_ratio=cas.color_to_depth_ratio,
            camera_model=frame.camera_model,
            visualization_frame=parameters.visualization_frame,
            center_axes_on_bounding_box=parameters.center_axes_on_bounding_box,
            show_object_mesh=parameters.visualize_object_mesh,
        )
        for estimated_pose in estimated_poses:
            visualization.draw_mesh(
                self.find_class_mesh(estimated_pose.request).shape,
                estimated_pose.object_pose,
                estimated_pose.label,
            )
        if cas.cloud is not None:
            visualization.geometries.append(NamedGeometry(CASViews.CLOUD, cas.cloud))
        output = self.get_annotator_output_struct()
        output.set_image(visualization.color_image)
        output.set_geometries(
            [geometry.to_drawable() for geometry in visualization.geometries]
        )
