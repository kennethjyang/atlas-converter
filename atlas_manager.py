"""Atlas access and instance manager.

Operations related to exposing access to Brain Globe atlases.
"""

from json import dump
from functools import cache
from pathlib import Path
from typing import Iterator

from brainglobe_atlasapi import list_atlases
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

from models import AtlasStructure, PinpointAtlasMetadata


@cache
def all_atlas_names() -> list[str]:
    """Returns sorted list of all latest BrainGlobe atlas names. Cached to avoid re-fetching."""
    return sorted(list_atlases.get_atlases_lastversions().keys())


def all_atlases() -> Iterator[BrainGlobeAtlas]:
    """Return all atlases."""
    yield from (
        # pyrefly: ignore [bad-argument-type]
        BrainGlobeAtlas(atlas, check_latest=True)
        for atlas in all_atlas_names()
    )


def allen_mouse_atlases() -> Iterator[BrainGlobeAtlas]:
    """Return only Allen Mouse atlases."""
    yield from (
        # pyrefly: ignore [bad-argument-type]
        BrainGlobeAtlas(atlas, check_latest=True)
        for atlas in all_atlas_names()
        if "allen_mouse" in atlas
    )


def pinpoint_atlases_root() -> Path:
    """Returns the output root for all atlases."""
    return Path.home() / "pinpoint_atlases"


def atlas_root_by_atlas(atlas: BrainGlobeAtlas) -> Path:
    """Returns the output root for this atlas.

    Args:
        atlas: Brain Globe atlas to return the output root path for.
    """
    return pinpoint_atlases_root() / atlas.metadata["name"]


def atlas_root_by_name(atlas_name: str) -> Path:
    """Returns the output root for this atlas (by name).

    Args:
        atlas_name: Name of the atlas to return the output root path for.
    """
    return pinpoint_atlases_root() / atlas_name


def ensure_path(file: Path) -> Path:
    """Returns the file path after creating the path if it doesn't exist.

    Args:
        file: Path to a file to write.
    """
    file.parent.mkdir(parents=True, exist_ok=True)
    return file


def pinpoint_atlas_metadata_for_group(
    group: list[BrainGlobeAtlas], remapped_structures: list[AtlasStructure | None]
) -> PinpointAtlasMetadata:
    """Return Pinpoint Atlas metadata for a given atlas group.

    Args:
        group: Group of BrainGlobe atlases to build a Pinpoint Atlas definition for.
        remapped_structures: Atlas structure LUT.
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
        structures=remapped_structures,
    )


def save_pinpoint_atlas_metadata(metadata: PinpointAtlasMetadata):
    """Write Pinpoint Atlas metadata to disk.

    Creates folders if needed.

    Args:
        metadata: Pinpoint Atlas metadata to write.
    """
    with open(ensure_path(atlas_root_by_name(metadata.name) / "atlas.json"), "w") as f:
        f.write(metadata.model_dump_json())


def save_pinpoint_atlas_metadata_schema():
    """Write Pinpoint Atlas model schema file to output root."""
    with open(ensure_path(pinpoint_atlases_root() / "atlas_schema.json"), "w") as f:
        dump(PinpointAtlasMetadata.model_json_schema(), f, separators=(",", ":"))
