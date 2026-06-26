from brainglobe_atlasapi import BrainGlobeAtlas
from numpy import dtype, ndarray, uint16
from pandas import Categorical

from atlas_manager import get_sorted_structure_ids


def remap_annotation_ids(
    atlas: BrainGlobeAtlas,
) -> ndarray[tuple[int, int, int], dtype[uint16]]:
    """Returns remap annotation values to the sorted structure IDs order.

    Args:
        atlas: Brain Globe atlas to remap the annotation of.
    """
    flat_atlas = atlas.annotation.ravel()
    remapped_flat = Categorical(
        flat_atlas, categories=get_sorted_structure_ids(atlas)
    ).codes.astype(uint16)
    return remapped_flat.reshape(atlas.shape)
