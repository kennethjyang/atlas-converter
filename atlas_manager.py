"""Atlas access and instance manager.

Operations related to exposing access to Brain Globe atlases.
"""

from json import dump
from pathlib import Path
from typing import Iterator

from brainglobe_atlasapi import list_atlases
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

from models import PinpointAtlasMetadata, StructureLut

"""Brain Globe atlas loading."""


def get_all_atlas_names_sorted() -> list[str]:
    """Returns sorted list of all latest BrainGlobe atlas names. Cached to avoid re-fetching."""
    return sorted(list_atlases.get_atlases_lastversions().keys())


def all_atlases() -> Iterator[BrainGlobeAtlas]:
    """Return all atlases."""
    yield from (
        # pyrefly: ignore [bad-argument-type]
        BrainGlobeAtlas(atlas, check_latest=True)
        for atlas in get_all_atlas_names_sorted()
    )


def allen_mouse_atlases() -> Iterator[BrainGlobeAtlas]:
    """Return only Allen Mouse atlases."""
    yield from (
        # pyrefly: ignore [bad-argument-type]
        BrainGlobeAtlas(atlas_name, check_latest=True)
        for atlas_name in get_all_atlas_names_sorted()
        if "allen_mouse" in atlas_name
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


def build_pinpoint_atlases_path() -> Path:
    """Returns the output root for all atlases."""
    return Path.home() / "pinpoint_atlases"


def build_atlas_path(atlas: str | BrainGlobeAtlas) -> Path:
    """Returns the output root for this atlas.

    Args:
        atlas: Atlas name or Brain Globe atlas to return the output root path for.
    """
    return build_pinpoint_atlases_path() / str(
        atlas if isinstance(atlas, str) else atlas.metadata["name"]
    )


def prepare_path(file: Path) -> Path:
    """Returns the file path after creating the path if it doesn't exist.

    Args:
        file: Path to a file to write.
    """
    file.parent.mkdir(parents=True, exist_ok=True)
    return file


def save_pinpoint_atlas_metadata(metadata: PinpointAtlasMetadata):
    """Write Pinpoint Atlas metadata to disk.

    Creates folders if needed.

    Args:
        metadata: Pinpoint Atlas metadata to write.
    """
    with open(prepare_path(build_atlas_path(metadata.name) / "atlas.json"), "w") as f:
        f.write(metadata.model_dump_json())


def save_pinpoint_atlas_metadata_schema():
    """Write Pinpoint Atlas model schema file to output root."""
    with open(
        prepare_path(build_pinpoint_atlases_path() / "atlas_schema.json"), "w"
    ) as f:
        dump(PinpointAtlasMetadata.model_json_schema(), f, separators=(",", ":"))
