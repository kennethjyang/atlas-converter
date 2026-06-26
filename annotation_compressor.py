"""Annotation compression operations."""

from typing import List

from brainglobe_atlasapi import BrainGlobeAtlas
from numpy import dtype, ndarray, searchsorted, uint16
from pandas import Categorical

from atlas_manager import sorted_structure_ids
from models import AtlasStructure


def remapped_annotation_ids(
    atlas: BrainGlobeAtlas,
) -> ndarray[tuple[int, int, int], dtype[uint16]]:
    """Returns remap annotation values to the sorted structure IDs order.

    Args:
        atlas: Brain Globe atlas to remap the annotation of.
    """
    flat_atlas = atlas.annotation.ravel()
    remapped_flat = Categorical(
        flat_atlas, categories=sorted_structure_ids(atlas)
    ).codes.astype(uint16)
    return remapped_flat.reshape(atlas.shape)


def remapped_structure_and_color_lut(
    atlas: BrainGlobeAtlas,
) -> tuple[list[AtlasStructure | None], list[int]]:
    """Returns structure LUT then color LUT for atlas following remapped IDs.

    Structure LUT is returned first followed by color LUT. Color values are the unsigned byte values from the atlas.

    Args:
        atlas: Brain Globe atlas to build a structure color LUT for.
    Raises:
        ValueError: if a structure is missing from the atlas hierarchy tree.
    """
    # Init LUTs with 0-structure as black (empty).
    structure_lut: List[AtlasStructure | None] = [None]
    color_lut = [0, 0, 0, 255]

    # Iterate through the structures skipping the 0-structure.
    ids = sorted_structure_ids(atlas)
    for structure_id in ids[1:]:
        # Get the structure data.
        structure_data = atlas.structures[structure_id]
        # pyrefly: ignore [bad-argument-type]
        hierarchy_node = atlas.hierarchy.get_node(structure_id)

        # Stop if the node is missing.
        if hierarchy_node is None:
            raise ValueError(f"Structure {structure_id} not found in hierarchy.")

        # Extract color into color LUT.
        color_lut.extend([*structure_data["rgb_triplet"], 255])

        # Extract parent and children.
        parent_og_id = hierarchy_node.predecessor(atlas.hierarchy.identifier)
        children_og_ids = hierarchy_node.successors(atlas.hierarchy.identifier)

        # Build structure (convert parent and children).
        structure_lut.append(
            AtlasStructure(
                name=structure_data["name"],
                acronym=structure_data["acronym"],
                parent_id=None
                if parent_og_id is None
                else searchsorted(ids, parent_og_id),
                children_ids=set(searchsorted(ids, children_og_ids)),
            )
        )

    return structure_lut, color_lut
