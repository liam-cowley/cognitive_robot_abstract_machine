"""
Tests for the doing-laundry analysis engine reading every frame for all its subtrees and
routing each of its queries to the subtree that answers it.
"""

from __future__ import annotations

# %% Imports
from unittest.mock import MagicMock

import numpy as np
import open3d
import pytest
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from rclpy.node import Node
from robokudo_msgs.action import Query

from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.laundry_grasp_pose import LaundryGraspPoseAnnotator
from robokudo.annotators.laundry_grasp_publisher import LaundryGraspPublisher
from robokudo.annotators.outputs import AnnotatorOutputs
from robokudo.annotators.query import GenerateQueryResult
from robokudo.annotators.steady_laundry_grasp import SteadyLaundryGraspAnnotator
from robokudo.annotators.open_vocabulary_object_detection_annotator import (
    OpenVocabularyObjectDetectionAnnotator,
    Owlv2ForObjectDetection,
    Owlv2Processor,
)
from robokudo.behaviours.query_stream import ContinueQueryStream, StartQueryStream
from robokudo.cas import CAS, CASViews
from robokudo.descriptors.analysis_engines.doing_laundry import (
    AnalysisEngine,
    DoingLaundryQuery,
)
from robokudo.pipeline import Pipeline
from robokudo.tree_components.query_based_task_scheduler import QueryBasedScheduler
from robokudo.types.annotation import Classification, PoseAnnotation
from robokudo.types.cv import Rect
from robokudo.types.laundry import LaundryBasketClassName
from robokudo.types.scene import ObjectHypothesis
from semantic_digital_twin.robots.hsrb import HSRBGripper
from semantic_digital_twin.world import World

from .dataset.laundry_basket_on_floor import LaundryBasketScene

# %% Fixtures


@pytest.fixture
def pipeline(node: Node, monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    """
    :return: The pipeline of the engine, built without loading the detection and
        segmentation models.
    """
    monkeypatch.setattr(Owlv2Processor, "from_pretrained", MagicMock())
    monkeypatch.setattr(Owlv2ForObjectDetection, "from_pretrained", MagicMock())
    monkeypatch.setattr(
        "robokudo.annotators.open_vocabulary_object_detection_annotator.SegmentAnythingModel",
        MagicMock(),
    )
    return AnalysisEngine().implementation()


@pytest.fixture
def scheduler(pipeline: Pipeline) -> QueryBasedScheduler:
    """
    :return: The query scheduler of the engine.
    """
    (scheduler,) = [
        behaviour
        for behaviour in pipeline.iterate()
        if isinstance(behaviour, QueryBasedScheduler)
    ]
    return scheduler


# %% Reading the camera


def test_the_pipeline_reads_and_preprocesses_every_frame_for_all_subtrees(
    pipeline: Pipeline, scheduler: QueryBasedScheduler
) -> None:
    """
    Each pipeline iteration reads one frame, so that its outputs are shown once the
    subtree of the query is done with it.
    """
    pipeline_child_types = {type(child) for child in pipeline.children}
    subtree_types = {
        type(behaviour)
        for subtree in scheduler.tasks.values()
        for behaviour in subtree.iterate()
    }

    assert {CollectionReaderAnnotator, ImagePreprocessorAnnotator} <= (
        pipeline_child_types
    )
    assert not {CollectionReaderAnnotator, ImagePreprocessorAnnotator} & subtree_types


# %% Routing


def test_a_tshirt_pile_query_streams_the_grasp(
    pipeline: Pipeline, scheduler: QueryBasedScheduler
) -> None:
    """
    The T-shirt pile subtree finds the grasp, holds it steady across the stream and
    publishes it, starting the stream the pipeline continues with in its next iteration.
    """
    route = scheduler.filter_fn(Query.Goal(query=DoingLaundryQuery.TSHIRT_PILE))

    subtree_behaviours = list(scheduler.tasks[route].iterate())
    (start,) = [
        behaviour
        for behaviour in subtree_behaviours
        if isinstance(behaviour, StartQueryStream)
    ]
    (continuation,) = [
        behaviour
        for behaviour in pipeline.iterate()
        if isinstance(behaviour, ContinueQueryStream)
    ]
    (steady,) = [
        behaviour
        for behaviour in subtree_behaviours
        if isinstance(behaviour, SteadyLaundryGraspAnnotator)
    ]
    assert {LaundryGraspPoseAnnotator, LaundryGraspPublisher} <= {
        type(behaviour) for behaviour in subtree_behaviours
    }
    assert start.query_stream is continuation.query_stream
    assert steady.query_stream is continuation.query_stream


def test_the_tshirt_pile_grasp_keeps_the_hsrb_hand_clear_of_the_walls(
    scheduler: QueryBasedScheduler,
) -> None:
    """
    The grasp keeps the open fingers of the robot in use, the HSRB, clear of the
    basket's walls.
    """
    route = scheduler.filter_fn(Query.Goal(query=DoingLaundryQuery.TSHIRT_PILE))

    (grasp_annotator,) = [
        behaviour
        for behaviour in scheduler.tasks[route].iterate()
        if isinstance(behaviour, LaundryGraspPoseAnnotator)
    ]
    assert (
        grasp_annotator.descriptor.parameters.gripper_maximum_opening
        == HSRBGripper.maximum_opening_width
    )


def test_a_find_basket_query_answers_once_with_the_pose_of_the_basket(
    scheduler: QueryBasedScheduler,
) -> None:
    """
    The find-basket subtree detects only laundry baskets, estimates their pose and
    answers the query with it, without streaming.
    """
    route = scheduler.filter_fn(Query.Goal(query=DoingLaundryQuery.FIND_BASKET))

    subtree_behaviours = list(scheduler.tasks[route].iterate())
    (detection,) = [
        behaviour
        for behaviour in subtree_behaviours
        if isinstance(behaviour, OpenVocabularyObjectDetectionAnnotator)
    ]
    subtree_types = {type(behaviour) for behaviour in subtree_behaviours}
    assert detection.descriptor.parameters.classes == list(LaundryBasketClassName)
    assert {ClusterPoseBBAnnotator, GenerateQueryResult} <= subtree_types
    assert StartQueryStream not in subtree_types


@pytest.fixture
def basket_scene(perception_world: World) -> LaundryBasketScene:
    """
    :return: A laundry basket seen from above as densely as the HSRB's head camera sees
        one, posed in RoboKudo's world.
    """
    return LaundryBasketScene(world_frame=perception_world.root, focal_length=500.0)


def basket_seen_by_find_basket_subtree(
    *,
    scheduler: QueryBasedScheduler,
    scene: LaundryBasketScene,
    basket_region: Rect,
    basket_mask: np.ndarray,
) -> ObjectHypothesis:
    """
    Run the find-basket subtree, but for its detector and its answer, on the scene's
    frame with a laundry basket the detector segmented with the given mask.

    :return: The basket hypothesis after every step of the subtree succeeded on it.
    """
    cas = scene.cas()
    cas.set(CASViews.CLOUD, open3d.geometry.PointCloud())
    basket = ObjectHypothesis()
    basket.roi.roi = basket_region
    basket.roi.mask = basket_mask
    basket.annotations.append(
        Classification(classname=LaundryBasketClassName.LAUNDRY_BASKET, confidence=0.9)
    )
    cas.annotations.append(basket)
    route = scheduler.filter_fn(Query.Goal(query=DoingLaundryQuery.FIND_BASKET))
    subtree = scheduler.tasks[route]
    pipeline = Pipeline()
    pipeline.add_child(subtree)
    pipeline.cas = cas
    Blackboard().get("annotator_output_pipeline_map_buffer").map[pipeline.name] = (
        AnnotatorOutputs()
    )
    for behaviour in subtree.children:
        if isinstance(
            behaviour, (OpenVocabularyObjectDetectionAnnotator, GenerateQueryResult)
        ):
            continue
        assert behaviour.update() == Status.SUCCESS
    pipeline.remove_child(subtree)
    return basket


@pytest.mark.usefixtures("annotator_output_buffer")
def test_a_find_basket_query_estimates_the_pose_from_the_depth_points_of_the_basket(
    scheduler: QueryBasedScheduler, basket_scene: LaundryBasketScene
) -> None:
    """
    The pose of a basket the detector segmented is estimated from the depth points its
    mask covers, centred on them.
    """
    basket = basket_seen_by_find_basket_subtree(
        scheduler=scheduler,
        scene=basket_scene,
        basket_region=basket_scene.basket_region,
        basket_mask=basket_scene.basket_mask(),
    )

    [pose] = CAS.filter_by_type(PoseAnnotation, basket.annotations)
    assert np.allclose(
        pose.translation,
        basket.points.get_axis_aligned_bounding_box().get_center(),
    )


@pytest.mark.usefixtures("annotator_output_buffer")
def test_a_find_basket_pose_ignores_the_floor_a_segmentation_spills_onto(
    scheduler: QueryBasedScheduler, basket_scene: LaundryBasketScene
) -> None:
    """
    Depth points of the floor beside the basket that its mask covers leave the pose
    where the basket's own points put it.
    """
    basket = basket_seen_by_find_basket_subtree(
        scheduler=scheduler,
        scene=basket_scene,
        basket_region=basket_scene.basket_region,
        basket_mask=basket_scene.basket_mask(),
    )
    spilled_basket = basket_seen_by_find_basket_subtree(
        scheduler=scheduler,
        scene=basket_scene,
        basket_region=basket_scene.spilled_basket_region(),
        basket_mask=basket_scene.spilled_basket_mask(),
    )

    [pose] = CAS.filter_by_type(PoseAnnotation, basket.annotations)
    [spilled_pose] = CAS.filter_by_type(PoseAnnotation, spilled_basket.annotations)
    assert np.allclose(spilled_pose.translation, pose.translation)


def test_every_query_of_the_challenge_has_a_subtree(
    scheduler: QueryBasedScheduler,
) -> None:
    """
    Each query name routes to a subtree of its own.
    """
    routes = {
        scheduler.filter_fn(Query.Goal(query=query_name))
        for query_name in DoingLaundryQuery
    }

    assert routes == set(scheduler.tasks)


def test_an_unknown_query_has_no_subtree(scheduler: QueryBasedScheduler) -> None:
    """
    A query the challenge does not know is left without a subtree, which aborts it.
    """
    route = scheduler.filter_fn(Query.Goal(query="dance"))

    assert route not in scheduler.tasks
