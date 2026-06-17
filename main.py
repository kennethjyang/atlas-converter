from math import ceil, sqrt
from pprint import pprint
from typing import List

from brainglobe_atlasapi import BrainGlobeAtlas, list_atlases
from brainglobe_atlasapi.list_atlases import get_all_atlases_lastversions
from numpy import arange, searchsorted, uint32, unique
from zarr import create_array
from zarr.codecs import BloscCodec, BloscShuffle

from pinpoint_atlas import AtlasStructure


def main():
    print("Loading Atlas.")
    atlas = BrainGlobeAtlas("allen_mouse_100um", check_latest=True)
    print("\tAtlas loaded.")
    print()

    # Remap structure IDs to consecutive IDs.
    print("Compacting ID range.")

    # Extract IDs.
    ids = unique(atlas.annotation)
    print("\tExtracted IDs and sort.")

    # Map from old ID's to consecutive range.
    remapped_annotation = searchsorted(ids, atlas.annotation)
    print("\tCompacted IDs.")
    print()

    # Create color and name LUT.
    print("Create color and name LUT for new IDs.")

    # Init color LUT with 0-structure as black (empty).
    color_lut = [0, 0, 0]
    structure_lut: List[AtlasStructure | None] = [None]

    # Iterate through the structures skipping the 0-structure.
    for structure_id in ids[1:]:
        # Get the structure data.
        structure_data = atlas.structures[structure_id]
        hierarchy_node = atlas.hierarchy.get_node(structure_id)

        # Crash out if the node is missing.
        if hierarchy_node is None:
            raise ValueError(f"Structure {structure_id} not found in hierarchy.")

        # Extract color into color LUT.
        color_lut.extend(structure_data["rgb_triplet"])

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
    print("\tMapped colors to LUT.")
    print()

    # Compress with Zarr.
    print("Compress Atlas into Zarr.")
    chunk_width = ceil(sqrt(1_000_000 / 4 / atlas.shape[1]))
    z = create_array(
        store=f"out/{atlas.metadata['name']}/{atlas.metadata['resolution'][0]}.zarr",
        shape=atlas.shape,
        chunks=(chunk_width, atlas.shape[1], chunk_width),
        shards=(chunk_width * 3, atlas.shape[1], chunk_width * 3),
        dtype=atlas.annotation.dtype,
        compressors=BloscCodec(shuffle=BloscShuffle.bitshuffle),
        overwrite=True,
    )
    z[:] = remapped_annotation


if __name__ == "__main__":
    main()
