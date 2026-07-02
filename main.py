"""Atlas converter pipeline for Pinpoint V

Build Pinpoint V compatible atlases from BrainGlobe-style atlases.
"""

from itertools import groupby
from pathlib import Path
from typing import Annotated, Iterator, Optional

from brainglobe_atlasapi import BrainGlobeAtlas
from rich.progress import Progress
from typer import Argument, Exit, Option, Typer

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
    get_converter_version,
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
        atlases: Iterator that yields BrainGlobe atlases.
        atlas_count: Total number of atlases from the generator.
        converted_atlases_path: Output directory and root for all atlases and atlas schema.
    """

    # Write the metadata schema.
    save_pinpoint_atlas_metadata_schema(converted_atlases_path)

    # Iterate through all BrainGlobe atlases.
    with Progress() as progress:
        task = progress.add_task("Converting BrainGlobe atlases...", total=atlas_count)

        for group_name, group_iterator in groupby(
            atlases, key=lambda lambda_atlas: lambda_atlas.metadata["name"]
        ):
            # Build output path for atlas.
            atlas_path = build_atlas_path(group_name, converted_atlases_path)

            # Get all atlases in the group.
            group = list(group_iterator)

            # Compress annotations and track group.
            for atlas in group[:-1]:
                save_annotation(atlas, atlas_path)
                progress.advance(task)

            # Get last atlas.
            last_atlas = group[-1]

            # Compress the last atlas and track progress after group finishes.
            save_annotation(last_atlas, atlas_path)

            # Build LUTs
            structure_lut = build_structure_lut(last_atlas)
            color_lut = build_color_lut(structure_lut)

            # Save metadata.
            save_pinpoint_atlas_metadata(
                build_pinpoint_atlas_metadata(group, structure_lut), atlas_path
            )

            # Save color LUT.
            save_color_lut(color_lut, atlas_path)

            # Save meshes.
            save_meshes(last_atlas, atlas_path)

            # Finish the group.
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
    """Convert all BrainGlobe formatted atlases found a specified directory."""
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


def print_version_callback(do_it: bool):
    """Callback to print the version and exit from CLI.

    Args:
        do_it: Flag to do the printing.
    """
    if do_it:
        print(get_converter_version())
        raise Exit()


@app.callback()
def callback(
    version: Annotated[
        Optional[bool],
        Option("--version", callback=print_version_callback, is_eager=True),
    ] = None,
):
    """Tool to convert BrainGlobe formatted atlases into Pinpoint V compatible atlases."""


if __name__ == "__main__":
    app()
