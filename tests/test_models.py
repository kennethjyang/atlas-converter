"""Tests for models.py."""

from typing import Any

import pytest
from pydantic import ValidationError

from models import (
    AtlasStructure,
    PinpointAtlasMetadata,
    ensure_sorted_and_unique,
    ensure_unique_structures,
)


def make_structure(**overrides: Any) -> AtlasStructure:
    defaults: dict[str, Any] = {
        "name": "Root",
        "acronym": "RT",
        "parent_id": None,
        "children_ids": frozenset(),
        "color": (10, 20, 30),
    }
    defaults.update(overrides)
    return AtlasStructure(**defaults)


class TestEnsureSortedAndUnique:
    def test_returns_sorted_tuple_by_min_value(self):
        assert ensure_sorted_and_unique(
            ((30.0, 30.0, 30.0), (10.0, 10.0, 10.0), (20.0, 20.0, 20.0))
        ) == ((10.0, 10.0, 10.0), (20.0, 20.0, 20.0), (30.0, 30.0, 30.0))

    def test_sorts_anisotropic_tuples_by_smallest_axis_value(self):
        assert ensure_sorted_and_unique(((25.0, 10.0, 10.0), (5.0, 25.0, 25.0))) == (
            (5.0, 25.0, 25.0),
            (25.0, 10.0, 10.0),
        )

    def test_already_sorted_returns_same_values(self):
        value = ((10.0, 10.0, 10.0), (20.0, 20.0, 20.0))
        assert ensure_sorted_and_unique(value) == value

    def test_raises_on_duplicate_values(self):
        with pytest.raises(ValueError, match="unique"):
            ensure_sorted_and_unique(((10.0, 10.0, 10.0), (10.0, 10.0, 10.0)))


class TestEnsureUniqueStructures:
    def test_returns_value_unchanged_when_unique(self):
        structures = (make_structure(name="A"), make_structure(name="B"))
        assert ensure_unique_structures(structures) == structures

    def test_raises_with_duplicate_positions_when_not_unique(self):
        duplicate = make_structure(name="A")
        structures = (duplicate, duplicate)
        with pytest.raises(ValueError, match="Duplicates"):
            ensure_unique_structures(structures)


class TestAtlasStructure:
    def test_valid_construction(self):
        structure = make_structure(
            name="Root",
            acronym="RT",
            parent_id=None,
            children_ids=frozenset({1, 2}),
            color=(1, 2, 3),
        )
        assert structure.name == "Root"
        assert structure.acronym == "RT"
        assert structure.parent_id is None
        assert structure.children_ids == frozenset({1, 2})
        assert structure.color == (1, 2, 3)

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            make_structure(name="")

    def test_empty_acronym_raises(self):
        with pytest.raises(ValidationError):
            make_structure(acronym="")

    def test_negative_parent_id_raises(self):
        with pytest.raises(ValidationError):
            make_structure(parent_id=-1)

    def test_parent_id_at_upper_bound_raises(self):
        with pytest.raises(ValidationError):
            make_structure(parent_id=1 << 16)

    def test_parent_id_within_range_is_accepted(self):
        structure = make_structure(parent_id=(1 << 16) - 1)
        assert structure.parent_id == (1 << 16) - 1

    def test_child_id_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            make_structure(children_ids=frozenset({1 << 16}))

    def test_color_component_below_zero_is_clamped(self):
        structure = make_structure(color=(-5, 10, 300))
        assert structure.color == (0, 10, 255)

    def test_color_component_above_255_is_clamped(self):
        structure = make_structure(color=(256, 256, 256))
        assert structure.color == (255, 255, 255)


class TestPinpointAtlasMetadata:
    def make_metadata(self, **overrides: Any) -> PinpointAtlasMetadata:
        defaults: dict[str, Any] = {
            "name": "test_atlas",
            "version": "1.0.0",
            "resolutions": ((25.0, 25.0, 25.0), (10.0, 10.0, 10.0)),
            "dimensions": (10.0, 8.0, 11.4),
            "root_id": 0,
            "structures": (make_structure(name="Root"),),
            "default_reference_coordinate": (1.0, 2.0, 3.0),
        }
        defaults.update(overrides)
        return PinpointAtlasMetadata(**defaults)

    def test_valid_construction(self):
        metadata = self.make_metadata()
        assert metadata.name == "test_atlas"
        assert metadata.resolutions == ((10.0, 10.0, 10.0), (25.0, 25.0, 25.0))
        assert metadata.dimensions == (10.0, 8.0, 11.4)
        assert metadata.default_reference_coordinate == (1.0, 2.0, 3.0)

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            self.make_metadata(name="")

    def test_empty_version_raises(self):
        with pytest.raises(ValidationError):
            self.make_metadata(version="")

    def test_unsorted_resolutions_are_sorted(self):
        metadata = self.make_metadata(
            resolutions=(
                (50.0, 50.0, 50.0),
                (10.0, 10.0, 10.0),
                (25.0, 25.0, 25.0),
            )
        )
        assert metadata.resolutions == (
            (10.0, 10.0, 10.0),
            (25.0, 25.0, 25.0),
            (50.0, 50.0, 50.0),
        )

    def test_duplicate_resolutions_raise(self):
        with pytest.raises(ValidationError):
            self.make_metadata(resolutions=((10.0, 10.0, 10.0), (10.0, 10.0, 10.0)))

    def test_root_id_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            self.make_metadata(root_id=1 << 16)

    def test_empty_structures_raises(self):
        with pytest.raises(ValidationError):
            self.make_metadata(structures=())

    def test_duplicate_structures_raise(self):
        structure = make_structure(name="Root")
        with pytest.raises(ValidationError):
            self.make_metadata(structures=(structure, structure))

    def test_serializes_with_camel_case_key(self):
        metadata = self.make_metadata(default_reference_coordinate=(5.0, 10.0, 15.0))
        dumped = metadata.model_dump()
        assert "defaultReferenceCoordinate" in dumped
        assert dumped["defaultReferenceCoordinate"] == (5.0, 10.0, 15.0)
