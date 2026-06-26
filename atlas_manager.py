"""Atlas access and instance manager.

Operations related to exposing access to Brain Globe atlases.
"""

from typing import Iterator

from brainglobe_atlasapi import list_atlases
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

# Latest list of all Brain Globe atlases
ALL_ATLAS_NAMES = sorted(list_atlases.get_atlases_lastversions().keys())


def all_atlases() -> Iterator[BrainGlobeAtlas]:
    """Return all atlases"""
    # pyrefly: ignore [bad-argument-type]
    yield from (BrainGlobeAtlas(atlas, check_latest=True) for atlas in ALL_ATLAS_NAMES)


def allen_mouse_atlases() -> Iterator[BrainGlobeAtlas]:
    """Return only Allen Mouse atlases"""
    yield from (
        # pyrefly: ignore [bad-argument-type]
        BrainGlobeAtlas(atlas, check_latest=True)
        for atlas in ALL_ATLAS_NAMES
        if "allen_mouse" in atlas
    )
