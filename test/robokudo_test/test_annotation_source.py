"""
Tests that annotations name the class of the annotator that produced them.
"""

from __future__ import annotations

# %% Imports

import numpy
import open3d
import pytest
from py_trees.common import Status

from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.annotators.shape_estimator import ShapeEstimatorAnnotator
from robokudo.annotators.static_object_detector import StaticObjectDetectorAnnotator
from robokudo.annotators.testing import EmptyAnnotator
from robokudo.cas import CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import Classification, SemanticColor, Shape
from robokudo.types.cv import Rect
from robokudo.types.scene import ObjectHypothesis


# %% Fixtures


@pytest.fixture
def renamed_annotator_name() -> str:
    """
    A custom annotator name that differs from every annotator class name.
    """
    return "RenamedAnnotator"


def _add_to_pipeline(annotator: BaseAnnotator) -> Pipeline:
    """
    :return: A set up pipeline whose only child is the given annotator.
    """
    pipeline = Pipeline()
    pipeline.add_child(annotator)
    pipeline.setup()
    return pipeline


# %% Class name


def test_class_name_is_readable_from_annotator_class() -> None:
    """
    The class name is available without instantiating the annotator.
    """
    assert EmptyAnnotator.get_class_name() == EmptyAnnotator.__name__


def test_class_name_ignores_custom_annotator_name(
    renamed_annotator_name: str,
) -> None:
    """
    A custom instance name does not change the class name.
    """
    annotator = EmptyAnnotator(name=renamed_annotator_name)

    assert annotator.get_class_name() == EmptyAnnotator.__name__


# %% Annotation source


def test_static_object_detector_classification_names_detector_class(
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    Classifications record the detector class as their source.
    """
    hypothesis = ObjectHypothesis()

    StaticObjectDetectorAnnotator.add_classification_annotation(
        object_hypothesis=hypothesis, class_name="cup"
    )

    (classification,) = hypothesis.annotations
    assert isinstance(classification, Classification)
    assert classification.source == StaticObjectDetectorAnnotator.get_class_name()


def test_renamed_cluster_color_annotator_names_its_class(
    renamed_annotator_name: str,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    Colors from a renamed annotator record its class, not its instance name.
    """
    height, width = 10, 10
    annotator = ClusterColorAnnotator(name=renamed_annotator_name)
    pipeline = _add_to_pipeline(annotator)
    color_image = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    hypothesis = ObjectHypothesis()
    hypothesis.roi.roi = Rect(width=width, height=height)
    hypothesis.roi.mask = numpy.full((height, width), 255, dtype=numpy.uint8)
    pipeline.cas.set(CASViews.CLOUD, open3d.geometry.PointCloud())
    pipeline.cas.set(CASViews.COLOR_IMAGE, color_image)
    pipeline.cas.annotations.append(hypothesis)

    assert annotator.update() == Status.SUCCESS

    colors = [
        annotation
        for annotation in hypothesis.annotations
        if isinstance(annotation, SemanticColor)
    ]
    assert len(colors) > 0
    assert all(
        color.source == ClusterColorAnnotator.get_class_name() for color in colors
    )


def test_renamed_shape_estimator_names_its_class(
    renamed_annotator_name: str,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    Shapes from a renamed annotator record its class, not its instance name.
    """
    annotator = ShapeEstimatorAnnotator(name=renamed_annotator_name)
    pipeline = _add_to_pipeline(annotator)
    box_surface = open3d.geometry.TriangleMesh.create_box(
        width=0.1, height=0.2, depth=0.3
    ).sample_points_uniformly(number_of_points=2000)
    hypothesis = ObjectHypothesis(points=box_surface)
    pipeline.cas.annotations.append(hypothesis)

    assert annotator.compute() == Status.SUCCESS

    (shape,) = [
        annotation
        for annotation in hypothesis.annotations
        if isinstance(annotation, Shape)
    ]
    assert shape.source == ShapeEstimatorAnnotator.get_class_name()
