"""Atlas converter pipeline for Pinpoint V

Build Pinpoint V compatible atlases from BrainGlobe-style atlases.
"""

from pathlib import Path
from typing import Annotated, Iterator

from brainglobe_atlasapi import BrainGlobeAtlas
from rich.progress import Progress
from typer import Argument, Typer

from atlas_compressor import (
    build_color_lut,
    build_structure_lut,
    save_annotation,
    save_color_lut,
    save_meshes,
)
from atlas_manager import (
    all_atlases,
    allen_mouse_atlases,
    build_atlas_path,
    build_default_converted_atlases_path,
    build_pinpoint_atlas_metadata,
    custom_atlases,
    get_all_allen_mouse_names_sorted,
    get_all_atlas_names_sorted,
    get_all_atlas_names_sorted_from,
    save_pinpoint_atlas_metadata,
    save_pinpoint_atlas_metadata_schema,
)

app = Typer()


def convert(
    atlases: Iterator[BrainGlobeAtlas],
    atlas_count: int,
    converted_atlases_path: Annotated[
        Path,
        Argument(help="Output directory and root for all atlases and atlas schema."),
    ] = build_default_converted_atlases_path(),
):
    """Convert atlases from an iterator and put them into the specified converted atlases' directory.

    Args:
        atlases: Generator that yields BrainGlobe atlases.
        atlas_count: Total number of atlases from the generator.
        converted_atlases_path: Output directory and root for all atlases and atlas schema.
    """

    # Write the metadata schema.
    save_pinpoint_atlas_metadata_schema(converted_atlases_path)

    # Remember seen atlas groups (to detect when to switch groups).
    groups: set[str] = set()

    # Current atlas group data.
    group: list[BrainGlobeAtlas] = []

    # Iterate through all BrainGlobe atlases.
    with Progress() as progress:
        task = progress.add_task("Converting BrainGlobe atlases...", total=atlas_count)
        for index, atlas in enumerate(atlases):
            # If this is the first atlas in a group or the very last one, finish off the previous group.
            atlas_group_name = atlas.metadata["name"]
            atlas_path = build_atlas_path(atlas_group_name, converted_atlases_path)
            if atlas_group_name not in groups or index == atlas_count - 1:
                # Create group if this was the last and only one.
                if len(group) == 0 and index == atlas_count - 1:
                    group = [atlas]

                # Finish the previous group.
                if len(group) > 0:
                    # Build LUTs.
                    structure_lut = build_structure_lut(atlas)
                    color_lut = build_color_lut(structure_lut)

                    # Save metadata.
                    save_pinpoint_atlas_metadata(
                        build_pinpoint_atlas_metadata(group, structure_lut), atlas_path
                    )

                    # Save color LUT.
                    save_color_lut(color_lut, atlas_path)

                    # Save meshes.
                    save_meshes(atlas, atlas_path)

                    # Reset current group.
                    group.clear()

                # Add self to processed groups.
                groups.add(atlas_group_name)

            # Compress annotation.
            save_annotation(atlas, atlas_path)

            # Save to group.
            group.append(atlas)
            progress.advance(task)


@app.command()
def brainglobe(
    converted_atlases_path: Annotated[
        Path,
        Argument(help="Output directory and root for all atlases and atlas schema."),
    ] = build_default_converted_atlases_path(),
):
    """Convert all BrainGlobe atlases."""
    convert(all_atlases(), len(get_all_atlas_names_sorted()), converted_atlases_path)


@app.command()
def custom(
    custom_atlases_path: Annotated[
        Path,
        Argument(help="Input directory of custom atlases in BrainGlobe atlas format."),
    ],
    converted_atlases_path: Annotated[
        Path,
        Argument(help="Output directory and root for all atlases and atlas schema."),
    ] = build_default_converted_atlases_path(),
):
    """Convert all BrainGlobe atlases."""
    convert(
        custom_atlases(custom_atlases_path),
        len(get_all_atlas_names_sorted_from(custom_atlases_path)),
        converted_atlases_path,
    )


@app.command()
def mouse(
    converted_atlases_path: Annotated[
        Path,
        Argument(help="Output directory and root for all atlases and atlas schema."),
    ] = build_default_converted_atlases_path(),
):
    """Convert only Allen CCF Mouse atlas from BrainGlobe.

    Used for testing. Will not update from online if the atlases were downloaded already.
    """
    convert(
        allen_mouse_atlases(),
        len(get_all_allen_mouse_names_sorted()),
        converted_atlases_path,
    )


if __name__ == "__main__":
    app()
