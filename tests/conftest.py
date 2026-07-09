"""Shared pytest fixtures."""

from typing import Callable, Optional
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

import atlas_compressor
import atlas_manager

MakeMockAtlas = Callable[..., MagicMock]


@pytest.fixture(autouse=True)
def _clear_caches():
    """Ensure the module-level caches don't leak state between tests."""
    atlas_manager.get_all_atlas_names_sorted.cache_clear()
    atlas_compressor.get_sorted_structure_ids.cache_clear()
    yield
    atlas_manager.get_all_atlas_names_sorted.cache_clear()
    atlas_compressor.get_sorted_structure_ids.cache_clear()


@pytest.fixture
def make_mock_atlas(mocker: MockerFixture) -> MakeMockAtlas:
    """Return a factory that builds a mock standing in for a BrainGlobeAtlas."""

    def _make(
        *,
        name: str = "test_atlas",
        resolution: tuple[int, ...] = (25, 25, 25),
        shape: tuple[int, ...] = (3, 3, 3),
        structures: Optional[dict[int, dict[str, object]]] = None,
        root_id: Optional[int] = 1,
        parents: Optional[dict[int, Optional[int]]] = None,
        children: Optional[dict[int, list[int]]] = None,
        annotation: Optional[object] = None,
        atlas_name: str = "test_atlas_25um",
        meshfile_map: Optional[dict[int, str]] = None,
    ) -> MagicMock:
        atlas = mocker.MagicMock()
        atlas.metadata = {"name": name, "resolution": resolution}
        atlas.shape = shape
        atlas.atlas_name = atlas_name
        resolved_structures: dict[int, dict[str, object]] = (
            structures if structures is not None else {}
        )
        atlas.structures = resolved_structures

        resolved_parents: dict[int, Optional[int]] = parents or {}
        resolved_children: dict[int, list[int]] = children or {}

        hierarchy = mocker.MagicMock()
        hierarchy.root = root_id
        hierarchy.identifier = "identifier"

        def get_node(structure_id: int) -> Optional[MagicMock]:
            if structure_id not in resolved_structures:
                return None
            node = mocker.MagicMock()
            node.predecessor.return_value = resolved_parents.get(structure_id)
            node.successors.return_value = resolved_children.get(structure_id, [])
            return node

        hierarchy.get_node.side_effect = get_node
        atlas.hierarchy = hierarchy

        if annotation is not None:
            atlas.annotation = annotation

        resolved_meshfile_map: dict[int, str] = meshfile_map or {}
        atlas.meshfile_from_structure.side_effect = lambda structure_id: (
            resolved_meshfile_map.get(structure_id, f"/mesh/{structure_id}.obj")
        )

        return atlas

    return _make
