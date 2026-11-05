"""
The meshes of object classes, for annotators that work on known object shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy
import trimesh
from semantic_digital_twin.world_description.geometry import Mesh
from typing_extensions import TYPE_CHECKING, Self

from robokudo.annotators.object_pose_estimation.exceptions import (
    DuplicateObjectClassMesh,
)

if TYPE_CHECKING:
    from pathlib import Path


# %% Object class meshes


@dataclass
class ObjectClassMesh:
    """
    The mesh showing the objects of one class.
    """

    class_name: str
    """
    Class name of the :class:`robokudo.types.annotation.Classification` the objects are
    detected with.
    """

    shape: Mesh
    """
    The mesh, whose :attr:`Mesh.scale` brings file coordinates without a declared unit
    to meters.
    """

    @cached_property
    def estimator_mesh_file(self) -> Path:
        """
        File holding exactly the mesh of :attr:`shape`, as one mesh in meters.
        """
        source_file = self.shape.local_file
        source_scene = trimesh.load_scene(source_file, process=False)
        if len(source_scene.graph.nodes_geometry) == 1 and numpy.allclose(
            source_scene.bounds, self.shape.mesh.bounds
        ):
            return source_file
        return Mesh.from_trimesh(self.shape.mesh).local_file


@dataclass
class ObjectClassMeshes:
    """
    The meshes of the object classes to pose, at most one per class.
    """

    meshes_by_class_name: dict[str, ObjectClassMesh] = field(default_factory=dict)
    """
    Meshes by the name of their class.
    """

    @classmethod
    def from_class_meshes(cls, class_meshes: list[ObjectClassMesh]) -> Self:
        """
        :param class_meshes: The meshes.
        :return: The meshes by the name of their class.
        :raises DuplicateObjectClassMesh: If a class has more than one mesh.
        """
        meshes_by_class_name = {}
        for class_mesh in class_meshes:
            if class_mesh.class_name in meshes_by_class_name:
                raise DuplicateObjectClassMesh(class_name=class_mesh.class_name)
            meshes_by_class_name[class_mesh.class_name] = class_mesh
        return cls(meshes_by_class_name=meshes_by_class_name)

    @property
    def estimator_mesh_files(self) -> list[Path]:
        """
        The file of every mesh, to hand to a pose estimator.
        """
        return [
            class_mesh.estimator_mesh_file
            for class_mesh in self.meshes_by_class_name.values()
        ]
