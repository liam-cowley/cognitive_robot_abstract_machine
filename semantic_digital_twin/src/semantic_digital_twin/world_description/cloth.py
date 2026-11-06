"""
Cloth a physical simulation deforms, and the garments it is made into.

A physics engine reads a :class:`ClothSheet` off the world as a simulator property; the
world's kinematic structure holds no cloth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from importlib.resources import files
from pathlib import Path

import trimesh

from krrood.patterns.path_enumeration import PathEnumeration
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.mixin import SimulatorAdditionalProperty
from semantic_digital_twin.world_description.geometry import Color, Mesh

# %% cloth sheets


@dataclass
class ClothElasticity:
    """
    How stiffly a cloth's material resists bending, as the elastic constants of a thin
    shell.
    """

    young_modulus: float = 3e5
    """
    Young's modulus of the material, in pascals.
    """

    poisson_ratio: float = 0.0
    """
    Poisson's ratio of the material.
    """


@dataclass
class Fabric:
    """
    A kind of cloth, with the physical values a simulation of it depends on.
    """

    elasticity: ClothElasticity
    """
    How stiffly the fabric resists bending.
    """

    thickness: float
    """
    Thickness of the fabric, in metres, which its bending stiffness grows with.
    """

    edge_damping: float
    """
    How strongly the edges of a sheet of the fabric damp their stretching.
    """


@dataclass
class Cotton(Fabric):
    """
    Cotton, with the values RGBench states for its T-shirt.
    """

    elasticity: ClothElasticity = field(
        default_factory=lambda: ClothElasticity(young_modulus=2.3e5, poisson_ratio=0.35)
    )
    """
    How stiffly cotton resists bending.
    """

    thickness: float = 3e-3
    """
    Thickness of cotton, in metres.
    """

    edge_damping: float = 0.1
    """
    How strongly the edges of a sheet of cotton damp their stretching.
    """


@dataclass
class ClothSheet(SimulatorAdditionalProperty):
    """
    A piece of cloth a physical simulation deforms: a sheet of triangles whose edges keep
    their length, which bends elastically and collides with everything but itself.
    """

    name: str
    """
    The name the simulation knows the cloth by.
    """

    surface: Mesh
    """
    The cloth's triangles, whose origin places them in the world frame when the
    simulation starts.
    """

    mass: float
    """
    Mass of the whole cloth, in kilograms.
    """

    fabric: Fabric = field(default_factory=Cotton)
    """
    The kind of cloth the sheet is made of.
    """

    collision_radius: float = 5e-4
    """
    How far from its surface the cloth touches what it collides with, in metres.
    """

    color: Color = field(default_factory=Color)
    """
    The color the cloth is drawn in.
    """

    @cached_property
    def starting_surface(self) -> trimesh.Trimesh:
        """
        :return: The cloth's triangles as they lie when the simulation starts, with their
            vertices in the world frame and its panels sewn together at their seams.
        """
        surface = self.surface.mesh.copy()
        surface.merge_vertices(merge_tex=True, merge_norm=True)
        return surface.apply_transform(self.surface.origin.to_np())


# %% garments


class GarmentFile(PathEnumeration):
    """
    The garments under the semantic_digital_twin resources, each a mesh of its panels
    lying flat as the garment was designed, the panels sharing the vertices of their
    seams.
    """

    T_SHIRT = (
        Path(files("semantic_digital_twin")).parent.parent
        / "resources"
        / "obj"
        / "t_shirt.obj"
    )
    """
    A short-sleeved T-shirt of two layers and 500 vertices, open at the hem, the neck
    and the cuffs, with its neckline at the origin, its hem along the x-axis and its
    sleeves along the y-axis. It is RGBench's green T-shirt, remeshed, see
    ``t_shirt_ATTRIBUTION.md`` next to it.
    """

    def create_surface(self, world_T_garment: HomogeneousTransformationMatrix) -> Mesh:
        """
        :param world_T_garment: Where the garment lies.
        :return: The garment as one sheet of triangles, placed in the world frame.
        """
        return Mesh(filename=str(self), origin=world_T_garment)
