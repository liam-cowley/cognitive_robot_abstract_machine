"""
Tests for the T-shirt: a perceivable, non-rigid object made of a kind of fabric.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.reasoning.queries import annotation_class_by_label
from semantic_digital_twin.semantic_annotations.mixins import (
    IsNonRigid,
    IsPerceivable,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import TShirt
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.cloth import Fabric

# %% fixtures


@pytest.fixture
def t_shirt() -> TShirt:
    """
    :return: A T-shirt in a world of its own.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        return TShirt.create_with_new_body_in_world(name="t_shirt", world=world)


# %% what a T-shirt is


def test_a_t_shirt_is_non_rigid(t_shirt: TShirt) -> None:
    """
    A T-shirt is declared as changing its shape.
    """
    assert isinstance(t_shirt, IsNonRigid)


def test_a_t_shirt_is_made_of_cotton(t_shirt: TShirt) -> None:
    """
    A T-shirt is made of cotton unless stated otherwise.
    """
    assert t_shirt.fabric is Fabric.COTTON


# %% perception


def test_a_t_shirt_is_perceivable(t_shirt: TShirt) -> None:
    """
    A T-shirt is declared as something perception can report.
    """
    assert isinstance(t_shirt, IsPerceivable)


def test_a_perception_label_naming_a_t_shirt_resolves_to_the_annotation() -> None:
    """
    A label from perception holding the class name is resolved to the T-shirt.
    """
    assert annotation_class_by_label("tshirt_white") is TShirt


# %% persistence


def test_a_t_shirt_keeps_its_fabric_after_a_database_roundtrip(
    t_shirt: TShirt, in_memory_session_maker: sessionmaker[Session]
) -> None:
    """
    The fabric is stored with the T-shirt and restored from the database.
    """
    stored_t_shirt = to_dao(t_shirt)
    with in_memory_session_maker() as session:
        session.add(stored_t_shirt)
        session.commit()
        session.expunge_all()
        restored = session.scalar(select(type(stored_t_shirt))).from_dao()

    assert restored.fabric is t_shirt.fabric


def test_a_t_shirt_keeps_its_fabric_after_a_json_roundtrip(t_shirt: TShirt) -> None:
    """
    The fabric travels with the T-shirt when it is serialized, as worlds do to
    synchronize.
    """
    serialized = t_shirt.to_json()
    tracker = WorldEntityWithIDKwargsTracker.from_world(t_shirt._world)

    restored = TShirt.from_json(serialized, **tracker.create_kwargs())

    assert restored.fabric is t_shirt.fabric
