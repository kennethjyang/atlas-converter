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
    return sorted(list_atlases.get_all_atlases_lastversions().keys())


def all_atlases() -> Iterator[BrainGlobeAtlas]:
    """Return all atlases."""
    yield from (
        # pyrefly: ignore [bad-argument-type]
        BrainGlobeAtlas(atlas)
        for atlas in get_all_atlas_names_sorted()
    )


@cache
def allen_mouse_atlases() -> list[BrainGlobeAtlas]:
    """Return only Allen CCF Mouse atlases."""
    atlas_resolutions = [10, 25, 50, 100]

    # Skip downloading if all atlases are already available locally.
    skip_check_latest = not all(
        f"allen_mouse_{resolution}um" in list_atlases.get_downloaded_atlases()
        for resolution in atlas_resolutions
    )

    return [
        # pyrefly: ignore [bad-argument-type]
        BrainGlobeAtlas(f"allen_mouse_{resolution}um", check_latest=skip_check_latest)
        for resolution in atlas_resolutions
    ]


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
