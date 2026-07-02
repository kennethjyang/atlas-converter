"""Atlas access and instance manager.

Operations related to exposing access to Brain Globe atlases.
"""

from functools import cache
from json import dump
from pathlib import Path
from typing import Iterator

from brainglobe_atlasapi import list_atlases
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

from models import PinpointAtlasMetadata, StructureLut

"""Brain Globe atlas loading."""


@cache
def get_all_atlas_names_sorted() -> list[str]:
    """Returns sorted list of all latest BrainGlobe atlas names. Cached to avoid re-fetching."""
    return sorted(list_atlases.get_atlases_lastversions().keys())


def get_all_atlas_names_sorted_from(custom_path: Path) -> list[str]:
    """Returns sorted list of all assumed BrainGlobe atlases in the specified directory.

    Atlas directories in the custom path are expected to follow the BrainGlobe naming convention.

    Args:
        custom_path: Path to custom atlas store directory.
    """
    return sorted(
        [
            directory.name.rsplit("_", 1)[0]
            for directory in custom_path.iterdir()
            if directory.is_dir()
        ]
    )


def get_all_allen_mouse_names_sorted() -> list[str]:
    """Returns sorted list of all Allen CCF Mouse atlas names."""
    return [f"allen_mouse_{resolution}um" for resolution in [100, 10, 25, 50]]


def all_atlases() -> Iterator[BrainGlobeAtlas]:
    """Return all atlases."""
    yield from (
        # pyrefly: ignore [bad-argument-type]
        BrainGlobeAtlas(atlas_name)
        for atlas_name in get_all_atlas_names_sorted()
    )


def custom_atlases(custom_path: Path) -> Iterator[BrainGlobeAtlas]:
    """Return atlases in a custom directory.

    Will continue show an exception for an erroneous atlas, but will not prevent the generator from continuing.

    Args:
        custom_path: Path to custom atlas store directory.
    """
    for atlas_name in get_all_atlas_names_sorted_from(custom_path):
        try:
            yield BrainGlobeAtlas(
                # pyrefly: ignore [bad-argument-type]
                atlas_name,
                brainglobe_dir=custom_path,
                check_latest=False,
            )
        except Exception as e:
            # Notify broken custom atlas, but don't stop the process.
            print(e)


def allen_mouse_atlases() -> Iterator[BrainGlobeAtlas]:
    """Return only Allen CCF Mouse atlases."""
    # Skip downloading if all atlases are already available locally.
    skip_check_latest = not all(
        atlas in list_atlases.get_downloaded_atlases()
        for atlas in get_all_allen_mouse_names_sorted()
    )

    yield from (
        # pyrefly: ignore [bad-argument-type]
        BrainGlobeAtlas(atlas_name, check_latest=skip_check_latest)
        for atlas_name in get_all_allen_mouse_names_sorted()
    )


"""Pinpoint Atlas metadata creation."""


def build_pinpoint_atlas_metadata(
    group: list[BrainGlobeAtlas], structure_lut: StructureLut
) -> PinpointAtlasMetadata:
    """Return Pinpoint Atlas metadata for a given atlas group.

    Args:
        group: Group of BrainGlobe atlases to build a Pinpoint Atlas definition for.
        structure_lut: Atlas structure LUT.
    Raises:
        ValueError: If the atlas group does not have a root node in the hierarchy.
    """
    # Extract first atlas for shared values.
    first_atlas = group[0]

    # Raise error of atlas doesn't have root.
    if first_atlas.hierarchy.root is None:
        raise ValueError(
            f'Root for atlas "{first_atlas.metadata["name"]}" not found in hierarchy!'
        )

    # Build output.
    return PinpointAtlasMetadata(
        name=first_atlas.metadata["name"],
        resolutions=[atlas.metadata["resolution"][0] for atlas in group],
        root_id=first_atlas.hierarchy.root,
        structures=structure_lut,
    )


"""File I/O."""


def build_default_converted_atlases_path() -> Path:
    """Returns the output root for all atlases."""
    return Path.home() / "pinpoint_atlases"


def build_atlas_path(
    atlas: str | BrainGlobeAtlas, converted_atlases_path: Path
) -> Path:
    """Returns the output root for this atlas.

    Args:
        atlas: Atlas name or Brain Globe atlas to return the output root path for.
        converted_atlases_path: Path to root directory for all converted atlases.
    """
    return converted_atlases_path / str(
        atlas if isinstance(atlas, str) else atlas.metadata["name"]
    )


def prepare_path(file: Path) -> Path:
    """Returns the file path after creating the path if it doesn't exist.

    Args:
        file: Path to a file to write.
    """
    file.parent.mkdir(parents=True, exist_ok=True)
    return file


def save_pinpoint_atlas_metadata(metadata: PinpointAtlasMetadata, atlas_path: Path):
    """Write Pinpoint Atlas metadata to disk.

    Creates folders if needed.

    Args:
        metadata: Pinpoint Atlas metadata to write.
        atlas_path: Output directory for this atlas.
    """
    with open(prepare_path(atlas_path / "atlas.json"), "w") as f:
        f.write(metadata.model_dump_json())


def save_pinpoint_atlas_metadata_schema(converted_atlases_path: Path):
    """Write Pinpoint Atlas model schema file to output root.

    Args:
        converted_atlases_path: Path to root directory for all converted atlases.
    """
    with open(prepare_path(converted_atlases_path / "atlas_schema.json"), "w") as f:
        dump(PinpointAtlasMetadata.model_json_schema(), f, separators=(",", ":"))
