"""Atlas access and instance manager.

Operations related to exposing access to Brain Globe atlases.
"""

from functools import cache
from pathlib import Path
from typing import Iterator

from brainglobe_atlasapi import list_atlases
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas


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
    """Returns the file path after creating the path if it doesn't exist."""
    file.parent.mkdir(parents=True, exist_ok=True)
    return file
