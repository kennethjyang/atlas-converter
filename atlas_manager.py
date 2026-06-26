"""Atlas access and instance manager.

Operations related to exposing access to Brain Globe atlases.
"""

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


def sorted_structure_ids(atlas: BrainGlobeAtlas):
    """Return all structure IDs in sorted order with 0 prepended.

    Args:
        atlas: The atlas to extract IDs from.

    Raises:
        ValueError: if the keys already uses 0 (reserved for "empty" space), or if IDs exceed 16-bit value.
    """
    sorted_keys = sorted(atlas.structures.keys())
    if 0 in sorted_keys:
        raise ValueError(
            "Atlas already uses reserved ID '0'. We assume this is kept free to indicate \"empty\" space."
        )

    if len(sorted_keys) + 1 >= (1 << 16):
        raise ValueError(
            f"Atlas structure IDs exceeds 16-bit data limit. We assume atlases with under {1 << 16} IDs."
        )

    return [0] + sorted_keys
