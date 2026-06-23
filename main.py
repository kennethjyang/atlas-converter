from json import dump
from math import ceil, sqrt
from os import makedirs
from pathlib import Path
from typing import List

from brainglobe_atlasapi import BrainGlobeAtlas
from numpy import searchsorted
from zarr import create_array
from zarr.codecs import BloscCodec, BloscShuffle

from pinpoint_atlas import AtlasStructure, PinpointAtlas


def main():
    print("Loading Atlas...")
    atlas = BrainGlobeAtlas("allen_mouse_10um", check_latest=True)
    print("\tAtlas loaded.")
    print()

    # Remap structure IDs to consecutive IDs.
    print("Compacting ID range...")

    # Extract IDs.
    ids = [0] + sorted(atlas.structures.keys())
    print("\tExtracted IDs and sort.")

    # Map from old ID's to consecutive range.
    remapped_annotation = searchsorted(ids, atlas.annotation)
    print("\tCompacted IDs.")
    print()

    # Create color and name LUT.
    print("Create color and name LUT for new IDs...")

    # Init color LUT with 0-structure as black (empty).
    color_lut = [0, 0, 0, 255]
    structure_lut: List[AtlasStructure | None] = [None]

    # Iterate through the structures skipping the 0-structure.
    for structure_id in ids[1:]:
        # Get the structure data.
        structure_data = atlas.structures[structure_id]
        # pyrefly: ignore [bad-argument-type]
        hierarchy_node = atlas.hierarchy.get_node(structure_id)

        # Crash out if the node is missing.
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
    print("\tCreated LUTs.")
    print()

    # Create output destination.
    atlas_root = Path.home() / "pinpoint_atlases" / atlas.metadata["name"]
    makedirs(atlas_root, exist_ok=True)

    # Compress annotations with Zarr.
    print("Compress annotation into Zarr...")
    chunk_width = ceil(sqrt(1_000_000 / 4 / atlas.shape[1]))
    annotation_zarr = create_array(
        store=atlas_root / f"{atlas.metadata['resolution'][0]}.zarr",
        shape=atlas.shape,
        chunks=(chunk_width, atlas.shape[1], chunk_width),
        shards=(chunk_width * 3, atlas.shape[1], chunk_width * 3),
        dtype=atlas.annotation.dtype,
        compressors=BloscCodec(shuffle=BloscShuffle.bitshuffle),
        overwrite=True,
    )
    annotation_zarr[:] = remapped_annotation
    print("\tCompressed annotations.")
    print()

    # Build atlas definition.
    print("Build Pinpoint atlas definition...")

    # Extract root ID.
    root_id = atlas.hierarchy.root
    if root_id is None:
        raise ValueError("Atlas root not found in hierarchy.")

    # Build atlas.
    pinpoint_atlas = PinpointAtlas(
        name=atlas.metadata["name"],
        resolutions=[atlas.metadata["resolution"][0]],
        root_id=root_id,
        structures=structure_lut,
    )
    print()

    print("Export Pinpoint atlas definition...")

    # Write atlas definition.
    with open(atlas_root / "atlas.json", "w") as f:
        f.write(pinpoint_atlas.model_dump_json())

    # Write LUT.
    with open(atlas_root / "lut.bin", "wb") as f:
        f.write(bytes(color_lut))

    # Write atlas schema (once in root).
    with open(atlas_root.parent / "atlas_schema.json", "w") as f:
        dump(PinpointAtlas.model_json_schema(), f, separators=(",", ":"))


if __name__ == "__main__":
    main()
