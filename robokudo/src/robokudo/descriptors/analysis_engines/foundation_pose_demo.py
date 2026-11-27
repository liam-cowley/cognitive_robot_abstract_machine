"""
Analysis engine posing the textured objects of the ROS package ``iai_objects`` with
FoundationPose in live RealSense data.

.. note::
    Needs the ROS package ``iai_objects`` and everything
    :mod:`robokudo.annotators.foundation_pose_estimator` needs.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from semantic_digital_twin.world_description.geometry import Mesh

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.foundation_pose import (
    FoundationPoseAnnotator,
    ObjectClassMesh,
)
from robokudo.annotators.foundation_pose_estimator import FoundationPoseEstimator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline
from robokudo.utils.file_loader import FileLoader

# %% Textured objects


class TexturedObject(StrEnum):
    """
    The objects of ``iai_objects`` that have a textured mesh, by the class name the YOLO
    model detects them as.

    A member's value is also the name of the object's directory in ``iai_objects``.
    """

    CHIPS_STACKED_CAN_ORIGINAL = "chips_stacked_can_original"
    CRACKER_CHEEZIT_BOX_ORIGINAL = "cracker_cheezit_box_original"
    MAYONNAISE_REMIA_BOTTLE = "mayonnaise_remia_bottle"
    MUESLI_VITALIS_BOX_NUTMIX = "muesli_vitalis_box_nutmix"
    MUSTARD_FRENCHS_BOTTLE = "mustard_frenchs_bottle"
    SODA_FANTA_CAN_ZERO = "soda_fanta_can_zero"
    SUGAR_DOMINO_BOX = "sugar_domino_box"
    TOMATOSOUP_CAMPBELLI_CAN = "tomatosoup_campbelli_can"

    @property
    def mesh_file(self) -> Path:
        """
        The textured mesh of the object, in meters.

        :raises OSError: If ``iai_objects`` or the mesh is not installed.
        """
        return FileLoader.get_path_to_file_in_ros_package(
            ros_pkg_name="iai_objects",
            relative_path=f"objects/{self.value}/textured.obj",
        )

    @property
    def class_mesh(self) -> ObjectClassMesh:
        """
        The mesh FoundationPose poses the detections of this object with.
        """
        return ObjectClassMesh(
            class_name=self.value, shape=Mesh(filename=str(self.mesh_file))
        )


# %% Analysis engine


class AnalysisEngine(AnalysisEngineInterface):
    """
    Detects the textured objects of ``iai_objects`` with the YOLO model of
    ``robokudo_suturo``, segments them with Segment Anything and estimates and tracks their 6D pose with FoundationPose.
    """

    def name(self) -> str:
        return "foundation_pose_demo"

    def implementation(self) -> Pipeline:
        detection_descriptor = YoloAnnotator.Descriptor()
        detection_descriptor.parameters.ros_package_name = "robokudo_suturo"
        detection_descriptor.parameters.weights_path = (
            "weights/ultimate_model_v11_1.26.4.pt"
        )
        detection_descriptor.parameters.class_names_path = (
            "data/json/id2name_ultimate_model_v11_1.26.4.json"
        )
        detection_descriptor.parameters.threshold = 0.6
        detection_descriptor.parameters.precision_mode = True

        pose_descriptor = FoundationPoseAnnotator.Descriptor()
        pose_descriptor.parameters.class_meshes = [
            textured_object.class_mesh for textured_object in TexturedObject
        ]
        pose_descriptor.parameters.visualize_object_mesh = True

        pipeline = Pipeline("FoundationPosePipeline")
        pipeline.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(
                    descriptor=CollectionReaderDescriptorFactory.create_descriptor(
                        "hsr_ros2"
                    )
                ),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                YoloAnnotator(descriptor=detection_descriptor),
                FoundationPoseAnnotator(
                    pose_estimator=FoundationPoseEstimator(),
                    descriptor=pose_descriptor,
                ),
            ]
        )
        return pipeline
