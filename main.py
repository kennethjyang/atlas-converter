from math import ceil, sqrt
from numpy import array, searchsorted
from pprint import pprint

from brainglobe_atlasapi import BrainGlobeAtlas
from zarr import create_array
from zarr.codecs import BloscCodec, BloscShuffle


def main():
    atlas = BrainGlobeAtlas("allen_mouse_10um", check_latest=True)

    # Remap structure ID's to consecutive identifiers from 1 onward.
    id_remap = {
        structure_id: index + 1
        for index, structure_id in enumerate(atlas.hierarchy.nodes.keys())
    }
    k = array(list(id_remap.values()))
    v = array(list(id_remap.keys()))

    sorted_values_index = v.argsort()
    k, v = k[sorted_values_index], v[sorted_values_index]

    mapped_index = searchsorted(k, atlas.annotation)
    remapped_annotation = v[mapped_index]
    pprint(remapped_annotation)


    # chunk_width = ceil(sqrt(1_000_000 / 4 / atlas.shape[1]))
    # z = create_array(
    #     store=f"out/{atlas.metadata['name']}/{atlas.metadata['resolution'][0]}.zarr",
    #     shape=atlas.shape,
    #     chunks=(chunk_width, atlas.shape[1], chunk_width),
    #     shards=(chunk_width * 3, atlas.shape[1], chunk_width * 3),
    #     dtype=atlas.annotation.dtype,
    #     compressors=BloscCodec(shuffle=BloscShuffle.bitshuffle),
    #     overwrite=True,
    # )
    # z[:] = atlas.annotation
    # print(z.info_complete())


if __name__ == "__main__":
    main()
