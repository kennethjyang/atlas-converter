from math import ceil, sqrt
from numpy import unique, arange, uint32, searchsorted
from pprint import pprint

from brainglobe_atlasapi import BrainGlobeAtlas
from zarr import create_array
from zarr.codecs import BloscCodec, BloscShuffle


def main():
    atlas = BrainGlobeAtlas("allen_mouse_100um", check_latest=True)
    print("Loaded atlas")

    # Remap structure ID's to consecutive identifiers from 1 onward.
    ids = unique(atlas.annotation)
    new_ids = arange(len(ids), dtype=uint32)
    print("Computed ids and new ones")
    mapped_index = searchsorted(ids, atlas.annotation)
    print("Create mapping from old to new")
    remapped_annotation = new_ids[mapped_index]

    # Compress with Zarr.
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
