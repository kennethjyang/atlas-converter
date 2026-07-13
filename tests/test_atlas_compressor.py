"""Tests for atlas_compressor.py."""

from math import ceil, sqrt
from pathlib import Path

import numpy as np
import pytest
from numpy import uint16
from pytest_mock import MockerFixture

from atlas_compressor import (
    build_color_lut,
    build_remapped_annotation,
    build_structure_lut,
    get_sorted_structure_ids,
    save_annotation,
    save_color_lut,
    save_meshes,
)
from models import AtlasStructure
from tests.conftest import MakeMockAtlas


class TestGetSortedStructureIds:
    def test_returns_sorted_keys(self, make_mock_atlas: MakeMockAtlas):
        atlas = make_mock_atlas(structures={5: {}, 2: {}, 9: {}})
        assert get_sorted_structure_ids(atlas) == [2, 5, 9]

    def test_raises_when_ids_exceed_16_bit_limit(self, make_mock_atlas: MakeMockAtlas):
        atlas = make_mock_atlas(structures={i: {} for i in range(65537)})
        with pytest.raises(ValueError, match="16-bit"):
            get_sorted_structure_ids(atlas)


class TestBuildRemappedAnnotation:
    def test_relabels_background_when_zero_is_a_valid_id(
        self, make_mock_atlas: MakeMockAtlas, mocker: MockerFixture
    ):
        ids = [0, 3, 7]
        annotation = np.array([[0, 3], [7, 0]], dtype=np.uint16)
        atlas = make_mock_atlas(shape=(2, 2), annotation=annotation)
        mocker.patch("atlas_compressor.get_sorted_structure_ids", return_value=ids)

        result = build_remapped_annotation(atlas)

        expected = np.array([[65535, 1], [2, 65535]], dtype=np.uint16)
        np.testing.assert_array_equal(result, expected)

    def test_no_relabel_when_zero_is_unused(
        self, make_mock_atlas: MakeMockAtlas, mocker: MockerFixture
    ):
        ids = [3, 7, 9]
        annotation = np.array([[0, 3], [7, 9]], dtype=np.uint16)
        atlas = make_mock_atlas(shape=(2, 2), annotation=annotation)
        mocker.patch("atlas_compressor.get_sorted_structure_ids", return_value=ids)

        result = build_remapped_annotation(atlas)

        expected = np.array([[65535, 0], [1, 2]], dtype=np.uint16)
        np.testing.assert_array_equal(result, expected)


class TestBuildStructureLut:
    def test_builds_lut_with_remapped_parent_and_children_ids(
        self, make_mock_atlas: MakeMockAtlas
    ):
        atlas = make_mock_atlas(
            structures={
                10: {"name": "Root", "acronym": "RT", "rgb_triplet": (1, 2, 3)},
                20: {"name": "Child", "acronym": "CH", "rgb_triplet": (4, 5, 6)},
            },
            parents={10: None, 20: 10},
            children={10: [20], 20: []},
        )

        lut = build_structure_lut(atlas)

        assert lut == (
            AtlasStructure(
                name="Root",
                acronym="RT",
                parent_id=None,
                children_ids=frozenset({1}),
                color=(1, 2, 3),
            ),
            AtlasStructure(
                name="Child",
                acronym="CH",
                parent_id=0,
                children_ids=frozenset(),
                color=(4, 5, 6),
            ),
        )

    def test_raises_when_hierarchy_node_missing(self, make_mock_atlas: MakeMockAtlas):
        atlas = make_mock_atlas(
            structures={5: {"name": "A", "acronym": "A", "rgb_triplet": (1, 2, 3)}}
        )
        atlas.hierarchy.get_node.side_effect = None
        atlas.hierarchy.get_node.return_value = None

        with pytest.raises(ValueError, match="not found in hierarchy"):
            build_structure_lut(atlas)


class TestBuildColorLut:
    def test_flattens_colors_with_trailing_alpha(self):
        structure_lut = (
            AtlasStructure(
                name="A",
                acronym="A",
                parent_id=None,
                children_ids=frozenset(),
                color=(1, 2, 3),
            ),
            AtlasStructure(
                name="B",
                acronym="B",
                parent_id=None,
                children_ids=frozenset(),
                color=(4, 5, 6),
            ),
        )

        assert build_color_lut(structure_lut) == [1, 2, 3, 255, 4, 5, 6, 255]


class TestSaveAnnotation:
    def test_creates_zarr_array_and_writes_remapped_annotation(
        self, make_mock_atlas: MakeMockAtlas, mocker: MockerFixture, tmp_path: Path
    ):
        atlas = make_mock_atlas(shape=(10, 20, 10), resolution=(25, 25, 25))
        mock_create_array = mocker.patch("atlas_compressor.create_array")
        remapped = mocker.MagicMock()
        mocker.patch(
            "atlas_compressor.build_remapped_annotation", return_value=remapped
        )

        save_annotation(atlas, tmp_path)

        chunk_width = ceil(sqrt(1_000_000 / 4 / atlas.shape[1]))
        _, kwargs = mock_create_array.call_args
        assert kwargs["store"] == tmp_path / "25.0-25.0-25.0.zarr"
        assert kwargs["shape"] == atlas.shape
        assert kwargs["chunks"] == (chunk_width, atlas.shape[1], chunk_width)
        assert kwargs["shards"] == (chunk_width * 3, atlas.shape[1], chunk_width * 3)
        assert kwargs["dtype"] == uint16
        assert kwargs["overwrite"] is True
        mock_create_array.return_value.__setitem__.assert_called_once_with(
            slice(None), remapped
        )


class TestSaveColorLut:
    def test_writes_bytes_to_expected_path(self, mocker: MockerFixture, tmp_path: Path):
        mock_open = mocker.patch("builtins.open", mocker.mock_open())

        save_color_lut([1, 2, 3, 255], tmp_path)

        mock_open.assert_called_once_with(tmp_path / "lut.bin", "wb")
        mock_open.return_value.write.assert_called_once_with(bytes([1, 2, 3, 255]))


class TestSaveMeshes:
    def test_maps_convert_over_items_excluding_first_sorted_id(
        self, make_mock_atlas: MakeMockAtlas, mocker: MockerFixture, tmp_path: Path
    ):
        atlas = make_mock_atlas(structures={1: {}, 2: {}, 3: {}})
        mock_pool_cls = mocker.patch("atlas_compressor.Pool")
        mock_pool = mock_pool_cls.return_value.__enter__.return_value

        save_meshes(atlas, tmp_path)

        mock_pool.map.assert_called_once()
        convert_func, items = mock_pool.map.call_args.args
        assert mock_pool.map.call_args.kwargs == {"chunksize": 4}
        assert items == [(1, "/mesh/2.obj"), (2, "/mesh/3.obj")]
        assert convert_func.keywords == {"atlas_path": tmp_path}

    def test_convert_mesh_skips_missing_mesh_file(
        self, make_mock_atlas: MakeMockAtlas, mocker: MockerFixture, tmp_path: Path
    ):
        atlas = make_mock_atlas(structures={1: {}, 2: {}})
        mock_pool_cls = mocker.patch("atlas_compressor.Pool")
        mock_pool = mock_pool_cls.return_value.__enter__.return_value
        mock_load_mesh = mocker.patch("atlas_compressor.load_mesh")
        mocker.patch("atlas_compressor.Path.is_file", return_value=False)

        save_meshes(atlas, tmp_path)
        convert_func, items = mock_pool.map.call_args.args
        convert_func(items[0])

        mock_load_mesh.assert_not_called()

    def test_convert_mesh_processes_and_exports_existing_mesh_file(
        self, make_mock_atlas: MakeMockAtlas, mocker: MockerFixture, tmp_path: Path
    ):
        atlas = make_mock_atlas(structures={1: {}, 2: {}})
        mock_pool_cls = mocker.patch("atlas_compressor.Pool")
        mock_pool = mock_pool_cls.return_value.__enter__.return_value
        mock_mesh = mocker.MagicMock()
        mock_mesh.simplify_quadric_decimation.return_value = mock_mesh
        mock_load_mesh = mocker.patch(
            "atlas_compressor.load_mesh", return_value=mock_mesh
        )
        mocker.patch("atlas_compressor.Path.is_file", return_value=True)

        save_meshes(atlas, tmp_path)
        convert_func, items = mock_pool.map.call_args.args
        convert_func(items[0])

        mock_load_mesh.assert_called_once_with(items[0][1])
        mock_mesh.simplify_quadric_decimation.assert_called_once_with(percent=0.95)
        mock_mesh.apply_scale.assert_called_once_with(0.001)
        mock_mesh.process.assert_called_once()
        mock_mesh.export.assert_called_once_with(
            tmp_path / "meshes" / f"{items[0][0]}.glb", include_normals=True
        )
