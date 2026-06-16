from math import ceil, sqrt
from pprint import pprint

from brainglobe_atlasapi import BrainGlobeAtlas
from numpy import arange, searchsorted, uint32, unique
from zarr import create_array
from zarr.codecs import BloscCodec, BloscShuffle


def main():
    print("1. Loading Atlas.")
    atlas = BrainGlobeAtlas("allen_mouse_100um", check_latest=True)
    print("\tAtlas loaded.")
    print()

    # Remap structure ID's to consecutive identifiers from 1 onward.
    print("2. Compacting ID range.")
    ids = unique(atlas.annotation)
    new_ids = arange(len(ids), dtype=uint32)
    print("\tExtracted IDs.")
    mapped_index = searchsorted(ids, atlas.annotation)
    print("\tCreated mapping from old IDs to new IDs.")
    remapped_annotation = new_ids[mapped_index]
    print("\tRemapped annotation to new IDs.")
    print()

    # Create LUT.
    print("3. Create LUT for colors to new ID.")
    lut = [0, 0, 0]
    for structure_id in ids[1:]:
        lut.extend(atlas.structures[structure_id]["rgb_triplet"])
    print("\tMapped colors to LUT.")
    print()

    # Compress with Zarr.
    print("4. Compress Atlas into Zarr.")
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
