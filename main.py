from math import ceil, sqrt
from pprint import pprint

from brainglobe_atlasapi import BrainGlobeAtlas
from zarr import create_array
from zarr.codecs import BloscCodec, BloscShuffle


def main():
    atlas = BrainGlobeAtlas("allen_mouse_10um", check_latest=True)
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
    z[:] = atlas.annotation
    print(z.info_complete())


if __name__ == "__main__":
    main()
