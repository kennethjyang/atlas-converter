"""Shared pytest fixtures."""

import pytest

import atlas_compressor
import atlas_manager


@pytest.fixture(autouse=True)
def _clear_caches():
    """Ensure the module-level caches don't leak state between tests."""
    atlas_manager.get_all_atlas_names_sorted.cache_clear()
    atlas_compressor.get_sorted_structure_ids.cache_clear()
    yield
    atlas_manager.get_all_atlas_names_sorted.cache_clear()
    atlas_compressor.get_sorted_structure_ids.cache_clear()


@pytest.fixture
def make_mock_atlas(mocker):
    """Return a factory that builds a mock standing in for a BrainGlobeAtlas."""

    def _make(
        *,
        name="test_atlas",
        resolution=(25, 25, 25),
        shape=(3, 3, 3),
        structures=None,
        root_id=1,
        parents=None,
        children=None,
        annotation=None,
        atlas_name="test_atlas_25um",
        meshfile_map=None,
    ):
        atlas = mocker.MagicMock()
        atlas.metadata = {"name": name, "resolution": resolution}
        atlas.shape = shape
        atlas.atlas_name = atlas_name
        atlas.structures = structures if structures is not None else {}

        parents = parents or {}
        children = children or {}

        hierarchy = mocker.MagicMock()
        hierarchy.root = root_id
        hierarchy.identifier = "identifier"

        def get_node(structure_id):
            if structure_id not in atlas.structures:
                return None
            node = mocker.MagicMock()
            node.predecessor.return_value = parents.get(structure_id)
            node.successors.return_value = children.get(structure_id, [])
            return node

        hierarchy.get_node.side_effect = get_node
        atlas.hierarchy = hierarchy

        if annotation is not None:
            atlas.annotation = annotation

        meshfile_map = meshfile_map or {}
        atlas.meshfile_from_structure.side_effect = lambda structure_id: (
            meshfile_map.get(structure_id, f"/mesh/{structure_id}.obj")
        )

        return atlas

    return _make
