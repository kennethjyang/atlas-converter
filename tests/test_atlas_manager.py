"""Tests for atlas_manager.py."""

from pathlib import Path


import atlas_manager
from atlas_manager import (
    all_atlases,
    allen_mouse_atlases,
    build_atlas_path,
    build_default_converted_atlases_path,
    custom_atlases,
    get_all_allen_mouse_names_sorted,
    get_all_atlas_names_sorted,
    get_all_atlas_names_sorted_from,
    prepare_path,
    save_pinpoint_atlas_metadata_schema,
)


class TestGetAllAtlasNamesSorted:
    def test_returns_sorted_names_from_remote_call(self, mocker):
        mocker.patch.object(
            atlas_manager.list_atlases,
            "get_all_atlases_lastversions",
            return_value={"zebra": "1", "apple": "1"},
        )
        assert get_all_atlas_names_sorted() == ["apple", "zebra"]

    def test_result_is_cached(self, mocker):
        mock_get = mocker.patch.object(
            atlas_manager.list_atlases,
            "get_all_atlases_lastversions",
            return_value={"apple": "1"},
        )
        get_all_atlas_names_sorted()
        get_all_atlas_names_sorted()
        assert mock_get.call_count == 1


class TestGetAllAtlasNamesSortedFrom:
    def test_returns_sorted_base_names_stripped_of_version(self, tmp_path):
        (tmp_path / "zebra_v2").mkdir()
        (tmp_path / "apple_v1").mkdir()
        assert get_all_atlas_names_sorted_from(tmp_path) == ["apple", "zebra"]

    def test_ignores_plain_files(self, tmp_path):
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
    def test_yields_one_atlas_per_sorted_name(self, mocker):
        mocker.patch(
            "atlas_manager.get_all_atlas_names_sorted",
            return_value=["apple", "zebra"],
        )
        mock_cls = mocker.patch("atlas_manager.BrainGlobeAtlas")

        result = list(all_atlases())

        assert mock_cls.call_args_list == [mocker.call("apple"), mocker.call("zebra")]
        assert result == [mock_cls.return_value, mock_cls.return_value]


class TestCustomAtlases:
    def test_yields_atlas_per_valid_name(self, mocker, tmp_path):
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
        self, mocker, tmp_path, capsys
    ):
        mocker.patch(
            "atlas_manager.get_all_atlas_names_sorted_from",
            return_value=["broken", "ok"],
        )
        good_atlas = mocker.MagicMock()

        def side_effect(name, **kwargs):
            if name == "broken":
                raise ValueError("boom")
            return good_atlas

        mocker.patch("atlas_manager.BrainGlobeAtlas", side_effect=side_effect)

        result = list(custom_atlases(tmp_path))

        assert result == [good_atlas]
        assert "boom" in capsys.readouterr().out


class TestAllenMouseAtlases:
    def test_skips_check_latest_when_all_downloaded(self, mocker):
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

    def test_checks_latest_when_atlas_missing_locally(self, mocker):
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
    def test_returns_path_under_home(self, mocker):
        mocker.patch.object(Path, "home", return_value=Path("/home/fakeuser"))
        assert build_default_converted_atlases_path() == Path(
            "/home/fakeuser/pinpoint_atlases"
        )


class TestBuildAtlasPath:
    def test_with_string_atlas_name(self):
        result = build_atlas_path("my_atlas", Path("/root"))
        assert result == Path("/root/my_atlas")

    def test_with_atlas_object(self, mocker):
        atlas = mocker.MagicMock()
        atlas.metadata = {"name": "my_atlas"}
        result = build_atlas_path(atlas, Path("/root"))
        assert result == Path("/root/my_atlas")


class TestPreparePath:
    def test_creates_missing_parent_directories(self, tmp_path):
        target = tmp_path / "nested" / "dir" / "file.txt"
        result = prepare_path(target)
        assert result == target
        assert target.parent.is_dir()

    def test_noop_when_parent_already_exists(self, tmp_path):
        target = tmp_path / "file.txt"
        result = prepare_path(target)
        assert result == target
        assert target.parent.is_dir()


class TestSavePinpointAtlasMetadataSchema:
    def test_writes_metadata_schema_to_expected_path(self, mocker, tmp_path):
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
