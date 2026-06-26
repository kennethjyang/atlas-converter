"""Annotation compression operations."""

from functools import lru_cache
from math import ceil, sqrt
from pathlib import Path
from typing import List

from brainglobe_atlasapi import BrainGlobeAtlas
from numpy import dtype, ndarray, searchsorted, uint16
from pandas import Categorical
from zarr import create_array
from zarr.codecs import BloscCodec, BloscShuffle

from atlas_manager import ensure_path, atlas_root_by_atlas
from models import AtlasStructure

type Annotation = ndarray[tuple[int, int, int], dtype[uint16]]


@lru_cache(1)
def sorted_structure_ids(atlas: BrainGlobeAtlas):
    """Return all structure IDs in sorted order with 0 prepended.

    The last call is cached.

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


def remapped_annotation_ids(
    atlas: BrainGlobeAtlas,
) -> Annotation:
    """Returns remap annotation values to the sorted structure IDs order.

    Args:
        atlas: Brain Globe atlas to remap the annotation of.
    """
    flat_atlas = atlas.annotation.ravel()
    remapped_flat = Categorical(
        flat_atlas, categories=sorted_structure_ids(atlas)
    ).codes.astype(uint16)
    return remapped_flat.reshape(atlas.shape)


@lru_cache(1)
def remapped_structure_and_color_lut(
    atlas: BrainGlobeAtlas,
) -> tuple[list[AtlasStructure | None], list[int]]:
    """Returns structure LUT then color LUT for atlas following remapped IDs.

    Structure LUT is returned first followed by color LUT. Color values are the unsigned byte values from the atlas.
    Last call is cached.

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


def compress_and_save_annotation(atlas: BrainGlobeAtlas):
    """Zarr compress an atlas's annotation volume and write it to disk.

    Args:
        atlas: BrainGlobe atlas to compress and save the annotation volume of.
    """
    chunk_width = ceil(sqrt(1_000_000 / 4 / atlas.shape[1]))
    annotation_zarr = create_array(
        store=ensure_path(
            atlas_root_by_atlas(atlas) / f"{atlas.metadata['resolution'][0]}.zarr"
        ),
        shape=atlas.shape,
        chunks=(chunk_width, atlas.shape[1], chunk_width),
        shards=(chunk_width * 3, atlas.shape[1], chunk_width * 3),
        dtype=uint16,
        compressors=BloscCodec(shuffle=BloscShuffle.bitshuffle),
        overwrite=True,
    )
    annotation_zarr[:] = remapped_annotation_ids(atlas)


def save_color_lut(lut: list[int], atlas_directory: Path):
    """Write color LUT as a binary array to disk.

    Args:
        lut: Color LUT to write to disk. All values must be unsigned bytes.
        atlas_directory: Output directory for this atlas.
    """
    with open(ensure_path(atlas_directory / "lut.bin"), "wb") as f:
        f.write(bytes(lut))
