"""Atlas converter pipeline for Pinpoint V

Build Pinpoint V compatible atlases from BrainGlobe-style atlases.
"""


from brainglobe_atlasapi import BrainGlobeAtlas

from annotation_compressor import (
    compress_and_save_annotation,
    remapped_structure_and_color_lut,
    save_color_lut,
)
from atlas_manager import (
    allen_mouse_atlases,
    atlas_root_by_atlas,
    pinpoint_atlas_metadata_for_group,
    save_pinpoint_atlas_metadata,
)
from models import (
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
        compress_and_save_annotation(atlas)
        atlas_group.append(atlas)

    # Compute structures and color LUTs.
    structure_lut, color_lut = remapped_structure_and_color_lut(atlas_group[0])

    # Build and save atlas metadata for group.
    save_pinpoint_atlas_metadata(
        pinpoint_atlas_metadata_for_group(atlas_group, structure_lut)
    )

    # Build and save structure color LUT.
    save_color_lut(
        color_lut,
        atlas_root_by_atlas(atlas_group[0]),
    )


if __name__ == "__main__":
    main()
