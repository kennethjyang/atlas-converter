"""Atlas access and instance manager.

Operations related to exposing access to Brain Globe atlases.
"""

from os import makedirs
from pathlib import Path
from typing import Iterator

from brainglobe_atlasapi import list_atlases
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

from annotation_compressor import remapped_structure_and_color_lut
from models import PinpointAtlas

# Latest list of all Brain Globe atlases
ALL_ATLAS_NAMES = sorted(list_atlases.get_atlases_lastversions().keys())


def all_atlases() -> Iterator[BrainGlobeAtlas]:
    """Return all atlases."""
    # pyrefly: ignore [bad-argument-type]
    yield from (BrainGlobeAtlas(atlas, check_latest=True) for atlas in ALL_ATLAS_NAMES)


def allen_mouse_atlases() -> Iterator[BrainGlobeAtlas]:
    """Return only Allen Mouse atlases."""
    yield from (
        # pyrefly: ignore [bad-argument-type]
        BrainGlobeAtlas(atlas, check_latest=True)
        for atlas in ALL_ATLAS_NAMES
        if "allen_mouse" in atlas
    )


def pinpoint_atlas_definition(atlas_group: list[BrainGlobeAtlas]) -> PinpointAtlas:
    """Return Pinpoint Atlas definition for a given atlas group.

    Args:
        atlas_group: Group of BrainGlobe atlases to build a Pinpoint Atlas definition for.
    Raises:
        ValueError: If the atlas group does not have a root node in the hierarchy.
    """
    # Extract first atlas for shared values.
    first_atlas = atlas_group[0]

    # Raise error of atlas doesn't have root.
    if first_atlas.hierarchy.root is None:
        raise ValueError(
            f'Root for atlas "{first_atlas.metadata["name"]}" not found in hierarchy!'
        )

    # Build output.
    return PinpointAtlas(
        name=first_atlas.metadata["name"],
        resolutions=[atlas.metadata["resolution"][0] for atlas in atlas_group],
        root_id=first_atlas.hierarchy.root,
        structures=remapped_structure_and_color_lut(first_atlas)[0],
    )


def pinpoint_atlases_root() -> Path:
    """Returns the output root for all atlases."""
    return Path.home() / "pinpoint_atlases"


def atlas_root(atlas: BrainGlobeAtlas) -> Path:
    """Returns the output root for this atlas.

    Args:
        atlas: Brain Globe atlas to return the output root address for.
    """
    return pinpoint_atlases_root() / atlas.metadata["name"]


def ensure_directory(path: Path) -> Path:
    """Returns the path and creates it on disk if it doesn't exist."""
    makedirs(path, exist_ok=True)
    return path
