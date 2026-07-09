"""Tests for main.py."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from typer import Exit

from main import (
    brainglobe,
    callback,
    convert,
    custom,
    get_converter_version,
    mouse,
    print_version_callback,
)


def make_atlas(
    mocker: MockerFixture, name: str, resolution: int, root: object = 1
) -> MagicMock:
    atlas = mocker.MagicMock()
    atlas.metadata = {"name": name, "resolution": (resolution,)}
    atlas.hierarchy.root = root
    return atlas


class TestGetConverterVersion:
    def test_returns_version_from_pyproject_toml(self, mocker: MockerFixture):
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch("main.load", return_value={"project": {"version": "9.9.9"}})

        assert get_converter_version() == "9.9.9"


class TestPrintVersionCallback:
    def test_prints_version_and_exits_when_do_it_true(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ):
        mocker.patch("main.get_converter_version", return_value="9.9.9")

        with pytest.raises(Exit):
            print_version_callback(True)

        assert capsys.readouterr().out.strip() == "9.9.9"

    def test_does_nothing_when_do_it_false(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ):
        mock_get_version = mocker.patch("main.get_converter_version")

        print_version_callback(False)

        mock_get_version.assert_not_called()
        assert capsys.readouterr().out == ""


class TestCallback:
    def test_does_not_raise(self):
        callback(None)


class TestConvert:
    def test_skips_group_with_no_root(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        mocker.patch("main.save_pinpoint_atlas_metadata_schema")
        mock_save_color_lut = mocker.patch("main.save_color_lut")
        mock_save_meshes = mocker.patch("main.save_meshes")
        mock_save_annotation = mocker.patch("main.save_annotation")
        atlas = make_atlas(mocker, "atlasA", 25, root=None)

        convert(iter([atlas]), 1, tmp_path)

        assert "atlasA is missing root" in capsys.readouterr().out
        mock_save_color_lut.assert_not_called()
        mock_save_meshes.assert_not_called()
        mock_save_annotation.assert_not_called()

    def test_full_happy_path_for_single_atlas_group(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        mocker.patch("main.save_pinpoint_atlas_metadata_schema")
        atlas_path = tmp_path / "atlasA"
        mocker.patch("main.build_atlas_path", return_value=atlas_path)
        mocker.patch("main.prepare_path", side_effect=lambda p: p)
        mocker.patch("main.build_structure_lut", return_value="structure_lut_sentinel")
        mocker.patch("main.build_color_lut", return_value="color_lut_sentinel")
        mock_save_color_lut = mocker.patch("main.save_color_lut")
        mock_save_meshes = mocker.patch("main.save_meshes")
        mock_save_annotation = mocker.patch("main.save_annotation")
        mocker.patch("main.get_sorted_structure_ids", return_value=[0, 5, 10])
        mocker.patch("main.get_converter_version", return_value="1.2.3")
        mock_metadata_cls = mocker.patch("main.PinpointAtlasMetadata")
        mock_metadata_cls.return_value.model_dump_json.return_value = '{"a":1}'
        mock_open = mocker.patch("builtins.open", mocker.mock_open())

        atlas = make_atlas(mocker, "atlasA", 25, root=5)

        convert(iter([atlas]), 1, tmp_path)

        mock_save_color_lut.assert_called_once_with("color_lut_sentinel", atlas_path)
        mock_save_meshes.assert_called_once_with(atlas, atlas_path)
        mock_save_annotation.assert_called_once_with(atlas, atlas_path)
        mock_metadata_cls.assert_called_once_with(
            name="atlasA",
            converter_version="1.2.3",
            resolutions=[25],
            root_id=1,
            structures="structure_lut_sentinel",
        )
        mock_open.assert_called_once_with(atlas_path / "atlas.json", "w")
        mock_open.return_value.write.assert_called_once_with('{"a":1}')

    def test_accumulates_resolutions_and_saves_annotation_for_each_atlas_in_group(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        mocker.patch("main.save_pinpoint_atlas_metadata_schema")
        atlas_path = tmp_path / "atlasA"
        mocker.patch("main.build_atlas_path", return_value=atlas_path)
        mocker.patch("main.prepare_path", side_effect=lambda p: p)
        mocker.patch("main.build_structure_lut", return_value="structure_lut_sentinel")
        mocker.patch("main.build_color_lut", return_value="color_lut_sentinel")
        mocker.patch("main.save_color_lut")
        mocker.patch("main.save_meshes")
        mock_save_annotation = mocker.patch("main.save_annotation")
        mocker.patch("main.get_sorted_structure_ids", return_value=[0, 5, 10])
        mocker.patch("main.get_converter_version", return_value="1.2.3")
        mock_metadata_cls = mocker.patch("main.PinpointAtlasMetadata")
        mock_metadata_cls.return_value.model_dump_json.return_value = "{}"
        mocker.patch("builtins.open", mocker.mock_open())

        atlas1 = make_atlas(mocker, "atlasA", 25, root=5)
        atlas2 = make_atlas(mocker, "atlasA", 10, root=5)

        convert(iter([atlas1, atlas2]), 2, tmp_path)

        assert mock_save_annotation.call_args_list == [
            mocker.call(atlas1, atlas_path),
            mocker.call(atlas2, atlas_path),
        ]
        assert mock_metadata_cls.call_args.kwargs["resolutions"] == [25, 10]


class TestBrainglobe:
    def test_calls_convert_with_all_atlases_and_count(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        mocker.patch("main.all_atlases", return_value=iter(["a", "b", "c"]))
        mocker.patch("main.get_all_atlas_names_sorted", return_value=["a", "b", "c"])
        mock_convert = mocker.patch("main.convert")

        brainglobe(tmp_path)

        mock_convert.assert_called_once()
        atlases_arg, count_arg, path_arg = mock_convert.call_args.args
        assert list(atlases_arg) == ["a", "b", "c"]
        assert count_arg == 3
        assert path_arg == tmp_path


class TestCustom:
    def test_calls_convert_with_custom_atlases_and_count(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        custom_path = tmp_path / "custom"
        mocker.patch("main.custom_atlases", return_value=iter(["x", "y"]))
        mocker.patch("main.get_all_atlas_names_sorted_from", return_value=["x", "y"])
        mock_convert = mocker.patch("main.convert")

        custom(custom_path, tmp_path)

        mock_convert.assert_called_once()
        atlases_arg, count_arg, path_arg = mock_convert.call_args.args
        assert list(atlases_arg) == ["x", "y"]
        assert count_arg == 2
        assert path_arg == tmp_path


class TestMouse:
    def test_calls_convert_with_allen_mouse_atlases_and_count(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        mocker.patch(
            "main.allen_mouse_atlases", return_value=iter(["m1", "m2", "m3", "m4"])
        )
        mocker.patch(
            "main.get_all_allen_mouse_names_sorted",
            return_value=["m1", "m2", "m3", "m4"],
        )
        mock_convert = mocker.patch("main.convert")

        mouse(tmp_path)

        mock_convert.assert_called_once()
        atlases_arg, count_arg, path_arg = mock_convert.call_args.args
        assert list(atlases_arg) == ["m1", "m2", "m3", "m4"]
        assert count_arg == 4
        assert path_arg == tmp_path
