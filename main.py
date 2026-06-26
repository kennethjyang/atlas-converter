"""Atlas converter pipeline for Pinpoint V

Build Pinpoint V compatible atlases from BrainGlobe-style atlases.
"""

from brainglobe_atlasapi import BrainGlobeAtlas

from atlas_compressor import (
    build_color_lut,
    build_structure_lut,
    save_annotation,
    save_color_lut,
)
from atlas_manager import (
    allen_mouse_atlases,
    build_atlas_path,
    build_pinpoint_atlas_metadata,
    save_pinpoint_atlas_metadata,
    save_pinpoint_atlas_metadata_schema,
)


def main():
    """Atlas Converter pipeline"""

    # Write the metadata schema.
    save_pinpoint_atlas_metadata_schema()

    # Build atlas group.
    atlas_group: list[BrainGlobeAtlas] = []

    # Iterate through atlases.
    for atlas in allen_mouse_atlases():
        print(f"Building {atlas.atlas_name}...")
        save_annotation(atlas)
        atlas_group.append(atlas)
        print("\tBuilt!")

    # Compute structures and color LUTs.
    structure_lut = build_structure_lut(atlas_group[0])
    color_lut = build_color_lut(structure_lut)

    # Build and save atlas metadata for group.
    save_pinpoint_atlas_metadata(
        build_pinpoint_atlas_metadata(atlas_group, structure_lut)
    )

    # Build and save structure color LUT.
    save_color_lut(
        color_lut,
        build_atlas_path(atlas_group[0]),
    )


if __name__ == "__main__":
    main()
