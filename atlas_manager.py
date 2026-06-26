"""Atlas access and instance manager.

Operations related to exposing access to Brain Globe atlases.
"""

from os import makedirs
from pathlib import Path
from typing import Iterator

from brainglobe_atlasapi import list_atlases
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

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


def ensure_directory(path: Path) -> Path:
    """Returns the path and creates it on disk if it doesn't exist."""
    makedirs(path, exist_ok=True)
    return path
