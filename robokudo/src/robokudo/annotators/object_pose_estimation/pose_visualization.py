"""
Drawing of posed objects into a color image and a 3D view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy
import open3d
from semantic_digital_twin.world_description.geometry import Color
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as numpy_typing
    from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
    from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel

    from robokudo.types.cv import ColorToDepthRatio


# %% 3D view geometries


class DrawableGeometryKey(StrEnum):
    """
    Keys of a named geometry in the 3D view, as :func:`open3d.visualization.draw` reads
    them.
    """

    NAME = "name"
    """
    Unique name of the geometry.
    """

    GEOMETRY = "geometry"
    """
    The geometry to draw.
    """


class GeometryKind(StrEnum):
    """
    What a geometry of the 3D view shows, ending its unique name.
    """

    AXES = "axes"
    """
    The coordinate axes of a pose.
    """

    BOUNDING_BOX = "bounding box"
    """
    The bounding box of an object.
    """

    OBJECT_MESH = "mesh"
    """
    The mesh of an object.
    """


@dataclass
class NamedGeometry:
    """
    A geometry of the 3D view with its unique name.
    """

    geometry_name: str
    """
    Unique name of the geometry.
    """

    geometry: open3d.geometry.Geometry3D
    """
    The geometry, in the camera frame at depth image resolution.
    """

    def to_drawable(
        self,
    ) -> dict[DrawableGeometryKey, str | open3d.geometry.Geometry3D]:
        """
        :return: The geometry in the form :func:`open3d.visualization.draw` reads.
        """
        return {
            DrawableGeometryKey.NAME: self.geometry_name,
            DrawableGeometryKey.GEOMETRY: self.geometry,
        }


# %% Visualization


@dataclass
class PoseVisualization:
    """
    The color image and the 3D geometries showing posed objects.
    """

    color_image: numpy_typing.NDArray[numpy.uint8]
    """
    Color image in BGR channel order, drawn into.
    """

    color_to_depth_ratio: ColorToDepthRatio
    """
    Size of the depth image relative to :attr:`color_image`.
    """

    camera_model: PinholeCameraModel
    """
    Calibration of the camera at depth image resolution.
    """

    axis_length: float = 0.1
    """
    Length of the drawn axes, in meters.
    """

    line_thickness: int = 2
    """
    Thickness of the lines drawn into :attr:`color_image`, in pixels.
    """

    label_font_scale: float = 0.6
    """
    Scale of the label font relative to its base size.
    """

    geometries: list[NamedGeometry] = field(default_factory=list)
    """
    Named geometries of the 3D view.
    """

    def add_axes_geometry(
        self, camera_T_axes: HomogeneousTransformationMatrix, label: str
    ) -> None:
        """
        Add the axes of a pose to the 3D view.

        :param camera_T_axes: Pose of the axes in the camera frame.
        :param label: Unique text identifying the object.
        """
        axes = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.axis_length
        )
        axes.transform(camera_T_axes.to_np())
        self.geometries.append(NamedGeometry(f"{label} {GeometryKind.AXES}", axes))

    def draw_bounding_box(
        self, bounding_box: open3d.geometry.OrientedBoundingBox, label: str
    ) -> None:
        """
        Draw the edges of a bounding box in front of the camera into the color image,
        labeled above its top left corner.

        :param bounding_box: Bounding box in the camera frame.
        :param label: Text drawn above the bounding box.
        """
        edges = open3d.geometry.LineSet.create_from_oriented_bounding_box(bounding_box)
        corners = numpy.asarray(edges.points)
        for start, end in numpy.asarray(edges.lines):
            self.draw_line(corners[start], corners[end], Color.GREEN())
        self.draw_label(corners, label)

    def draw_label(
        self, points: numpy_typing.NDArray[numpy.float64], label: str
    ) -> None:
        """
        Draw a label above the top left corner of the points in front of the camera.

        :param points: Points in the camera frame, one per row.
        :param label: Text to draw.
        """
        points_in_front = points[points[:, 2] > 0]
        if len(points_in_front) == 0:
            return
        left, top = self.color_to_depth_ratio.project_to_color_pixels(
            points_in_front, self.camera_model
        ).min(axis=0)
        cv2.putText(
            self.color_image,
            label,
            (int(left), int(top)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.label_font_scale,
            self.to_blue_green_red_bytes(Color.WHITE()),
            lineType=cv2.LINE_AA,
        )

    def draw_axes(self, camera_T_axes: HomogeneousTransformationMatrix) -> None:
        """
        Draw the x, y and z axes of a pose in red, green and blue into the color image,
        unless the end of an axis is not in front of the camera.

        :param camera_T_axes: Pose of the axes in the camera frame.
        """
        matrix = camera_T_axes.to_np()
        rotation, translation = matrix[:3, :3], matrix[:3, 3]
        axis_end_depths = translation[2] + rotation[2] * self.axis_length
        if translation[2] <= 0 or (axis_end_depths <= 0).any():
            return
        color_camera_model = self.camera_model.scaled(
            scale_x=1 / self.color_to_depth_ratio.horizontal,
            scale_y=1 / self.color_to_depth_ratio.vertical,
        )
        rotation_vector, _ = cv2.Rodrigues(rotation)
        cv2.drawFrameAxes(
            self.color_image,
            color_camera_model.intrinsic_matrix,
            None,
            rotation_vector,
            translation,
            self.axis_length,
            self.line_thickness,
        )

    def draw_line(
        self,
        start: numpy_typing.NDArray[numpy.float64],
        end: numpy_typing.NDArray[numpy.float64],
        color: Color,
    ) -> None:
        """
        Draw a line into the color image, unless an end of it is not in front of the
        camera.

        :param start: Point the line starts at, in the camera frame.
        :param end: Point the line ends at, in the camera frame.
        :param color: Color of the line.
        """
        if start[2] <= 0 or end[2] <= 0:
            return
        start_pixel, end_pixel = self.color_to_depth_ratio.project_to_color_pixels(
            numpy.array([start, end]), self.camera_model
        )
        cv2.line(
            self.color_image,
            (int(start_pixel[0]), int(start_pixel[1])),
            (int(end_pixel[0]), int(end_pixel[1])),
            self.to_blue_green_red_bytes(color),
            self.line_thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def to_blue_green_red_bytes(color: Color) -> tuple[int, int, int]:
        """
        :param color: A color.
        :return: The color as OpenCV byte channels in blue, green, red order.
        """
        maximum = Color.opencv_maximum_channel_value
        red, green, blue = color.to_rgb()
        return round(blue * maximum), round(green * maximum), round(red * maximum)
