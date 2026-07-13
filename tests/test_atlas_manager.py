"""Tests for atlas_manager.py."""

from pathlib import Path

import pytest
from pytest_mock import MockerFixture

import atlas_manager
from atlas_manager import (
    DEFAULT_REFERENCE_COORDINATE_OVERRIDES,
    all_atlases,
    allen_mouse_atlases,
    build_atlas_path,
    build_default_converted_atlases_path,
    build_default_reference_coordinate,
    custom_atlases,
    ensure_asr_orientation,
    get_all_allen_mouse_names_sorted,
    get_all_atlas_names_sorted,
    get_all_atlas_names_sorted_from,
    get_atlas_resolution,
    prepare_path,
    save_pinpoint_atlas_metadata_schema,
)
from tests.conftest import MakeMockAtlas


class TestGetAllAtlasNamesSorted:
    def test_returns_sorted_names_from_remote_call(self, mocker: MockerFixture):
        mocker.patch.object(
            atlas_manager.list_atlases,
            "get_all_atlases_lastversions",
            return_value={"zebra": "1", "apple": "1"},
        )
        assert get_all_atlas_names_sorted() == ["apple", "zebra"]

    def test_result_is_cached(self, mocker: MockerFixture):
        mock_get = mocker.patch.object(
            atlas_manager.list_atlases,
            "get_all_atlases_lastversions",
            return_value={"apple": "1"},
        )
        get_all_atlas_names_sorted()
        get_all_atlas_names_sorted()
        assert mock_get.call_count == 1


class TestGetAllAtlasNamesSortedFrom:
    def test_returns_sorted_base_names_stripped_of_version(self, tmp_path: Path):
        (tmp_path / "zebra_v2").mkdir()
        (tmp_path / "apple_v1").mkdir()
        assert get_all_atlas_names_sorted_from(tmp_path) == ["apple", "zebra"]

    def test_ignores_plain_files(self, tmp_path: Path):
        (tmp_path / "apple_v1").mkdir()
        (tmp_path / "not_a_dir.txt").write_text("data")
        assert get_all_atlas_names_sorted_from(tmp_path) == ["apple"]


class TestGetAllAllenMouseNamesSorted:
    def test_returns_expected_names(self):
        assert get_all_allen_mouse_names_sorted() == [
            "allen_mouse_100um",
            "allen_mouse_10um",
            "allen_mouse_25um",
            "allen_mouse_50um",
        ]


class TestAllAtlases:
    def test_yields_one_atlas_per_sorted_name(self, mocker: MockerFixture):
        mocker.patch(
            "atlas_manager.get_all_atlas_names_sorted",
            return_value=["apple", "zebra"],
        )
        mock_cls = mocker.patch("atlas_manager.BrainGlobeAtlas")

        result = list(all_atlases())

        assert mock_cls.call_args_list == [mocker.call("apple"), mocker.call("zebra")]
        assert result == [mock_cls.return_value, mock_cls.return_value]


class TestCustomAtlases:
    def test_yields_atlas_per_valid_name(self, mocker: MockerFixture, tmp_path: Path):
        mocker.patch(
            "atlas_manager.get_all_atlas_names_sorted_from",
            return_value=["apple", "zebra"],
        )
        mock_cls = mocker.patch("atlas_manager.BrainGlobeAtlas")

        result = list(custom_atlases(tmp_path))

        assert result == [mock_cls.return_value, mock_cls.return_value]
        mock_cls.assert_any_call("apple", brainglobe_dir=tmp_path, check_latest=False)
        mock_cls.assert_any_call("zebra", brainglobe_dir=tmp_path, check_latest=False)

    def test_prints_exception_and_continues_when_atlas_raises(
        self, mocker: MockerFixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        mocker.patch(
            "atlas_manager.get_all_atlas_names_sorted_from",
            return_value=["broken", "ok"],
        )
        good_atlas = mocker.MagicMock()

        def side_effect(name: str, **kwargs: object):
            if name == "broken":
                raise ValueError("boom")
            return good_atlas

        mocker.patch("atlas_manager.BrainGlobeAtlas", side_effect=side_effect)

        result = list(custom_atlases(tmp_path))

        assert result == [good_atlas]
        assert "boom" in capsys.readouterr().out


class TestAllenMouseAtlases:
    def test_skips_check_latest_when_all_downloaded(self, mocker: MockerFixture):
        mocker.patch.object(
            atlas_manager.list_atlases,
            "get_downloaded_atlases",
            return_value=[
                "allen_mouse_100um",
                "allen_mouse_10um",
                "allen_mouse_25um",
                "allen_mouse_50um",
            ],
        )
        mock_cls = mocker.patch("atlas_manager.BrainGlobeAtlas")

        list(allen_mouse_atlases())

        for call in mock_cls.call_args_list:
            assert call.kwargs["check_latest"] is False

    def test_checks_latest_when_atlas_missing_locally(self, mocker: MockerFixture):
        mocker.patch.object(
            atlas_manager.list_atlases,
            "get_downloaded_atlases",
            return_value=["allen_mouse_100um"],
        )
        mock_cls = mocker.patch("atlas_manager.BrainGlobeAtlas")

        list(allen_mouse_atlases())

        for call in mock_cls.call_args_list:
            assert call.kwargs["check_latest"] is True


class TestBuildDefaultConvertedAtlasesPath:
    def test_returns_path_under_home(self, mocker: MockerFixture):
        mocker.patch.object(Path, "home", return_value=Path("/home/fakeuser"))
        assert build_default_converted_atlases_path() == Path(
            "/home/fakeuser/pinpoint_atlases"
        )


class TestBuildAtlasPath:
    def test_with_string_atlas_name(self):
        result = build_atlas_path("my_atlas", Path("/root"))
        assert result == Path("/root/my_atlas")

    def test_with_atlas_object(self, mocker: MockerFixture):
        atlas = mocker.MagicMock()
        atlas.metadata = {"name": "my_atlas"}
        result = build_atlas_path(atlas, Path("/root"))
        assert result == Path("/root/my_atlas")


class TestPreparePath:
    def test_creates_missing_parent_directories(self, tmp_path: Path):
        target = tmp_path / "nested" / "dir" / "file.txt"
        result = prepare_path(target)
        assert result == target
        assert target.parent.is_dir()

    def test_noop_when_parent_already_exists(self, tmp_path: Path):
        target = tmp_path / "file.txt"
        result = prepare_path(target)
        assert result == target
        assert target.parent.is_dir()


class TestSavePinpointAtlasMetadataSchema:
    def test_writes_metadata_schema_to_expected_path(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        mock_dump = mocker.patch("atlas_manager.dump")
        mock_open = mocker.patch("builtins.open", mocker.mock_open())

        save_pinpoint_atlas_metadata_schema(tmp_path)

        expected_path = tmp_path / "atlas_schema.json"
        mock_open.assert_called_once_with(expected_path, "w")
        assert (
            mock_dump.call_args[0][0]
            == atlas_manager.PinpointAtlasMetadata.model_json_schema()
        )
        assert mock_dump.call_args[1] == {"separators": (",", ":")}


class TestGetAtlasResolution:
    def test_returns_resolution_as_float_tuple(self, make_mock_atlas: MakeMockAtlas):
        atlas = make_mock_atlas(resolution=(25, 10, 10))
        assert get_atlas_resolution(atlas) == (25.0, 10.0, 10.0)


class TestEnsureAsrOrientation:
    def test_does_not_raise_for_asr(self, make_mock_atlas: MakeMockAtlas):
        atlas = make_mock_atlas(orientation="asr")
        ensure_asr_orientation(atlas)

    def test_raises_for_non_asr(self, make_mock_atlas: MakeMockAtlas):
        atlas = make_mock_atlas(name="weird_atlas", orientation="lps")
        with pytest.raises(ValueError, match="weird_atlas"):
            ensure_asr_orientation(atlas)


class TestBuildDefaultReferenceCoordinate:
    def test_returns_override_when_name_matches(self, make_mock_atlas: MakeMockAtlas):
        atlas = make_mock_atlas(name="allen_mouse", shape=(1000, 500, 2000))
        result = build_default_reference_coordinate(atlas)
        assert result == (5.7, 0.44, 5.4)

    def test_computes_default_for_non_override_atlas(
        self, make_mock_atlas: MakeMockAtlas
    ):
        atlas = make_mock_atlas(
            name="test_atlas", shape=(200, 100, 400), resolution=(25, 25, 25)
        )
        result = build_default_reference_coordinate(atlas)
        # shape_um = (5000, 2500, 10000)
        # AP center (axis 0) = 5000 / 2 / 1000 = 2.5 mm
        # DV = 0.0
        # ML center (axis 2) = 10000 / 2 / 1000 = 5.0 mm
        assert result == (2.5, 0.0, 5.0)

    def test_computes_default_with_different_resolution(
        self, make_mock_atlas: MakeMockAtlas
    ):
        atlas = make_mock_atlas(
            name="another_atlas", shape=(100, 200, 300), resolution=(10, 10, 10)
        )
        result = build_default_reference_coordinate(atlas)
        # shape_um = (1000, 2000, 3000)
        # AP center = 1000 / 2 / 1000 = 0.5 mm
        # ML center = 3000 / 2 / 1000 = 1.5 mm
        assert result == (0.5, 0.0, 1.5)

    def test_overrides_dict_contains_allen_mouse(self):
        assert "allen_mouse" in DEFAULT_REFERENCE_COORDINATE_OVERRIDES
        assert DEFAULT_REFERENCE_COORDINATE_OVERRIDES["allen_mouse"] == (
            5.7,
            0.44,
            5.4,
        )
