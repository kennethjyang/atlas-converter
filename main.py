"""Atlas conversion pipeline for Pinpoint V

andles CLI, commands, and conversion pipeline.
"""

from itertools import groupby
from pathlib import Path
from tomllib import load
from typing import Annotated, Iterator, Optional

from brainglobe_atlasapi import BrainGlobeAtlas
from numpy import searchsorted
from rich.progress import Progress
from typer import Argument, Exit, Option, Typer

from atlas_compressor import (
    build_color_lut,
    build_structure_lut,
    get_sorted_structure_ids,
    save_annotation,
    save_color_lut,
    save_meshes,
)
from atlas_manager import (
    all_atlases,
    allen_mouse_atlases,
    build_atlas_path,
    build_default_converted_atlases_path,
    build_default_reference_coordinate,
    custom_atlases,
    get_all_allen_mouse_names_sorted,
    get_all_atlas_names_sorted,
    get_all_atlas_names_sorted_from,
    prepare_path,
    save_pinpoint_atlas_metadata_schema,
)
from models import PinpointAtlasMetadata

app = Typer()


def get_converter_version() -> str:
    """Return the version of the converter as specified in the pyproject.toml file."""
    with open(Path(__file__).parent / "pyproject.toml", "rb") as f:
        return load(f)["project"]["version"]


def print_version_callback(do_it: bool):
    """Callback to print the version and exit from CLI.

    Args:
        do_it: Flag to do the printing.
    """
    if do_it:
        print(get_converter_version())
        raise Exit()


# noinspection PyUnusedLocal
@app.callback()
def callback(
    version: Annotated[
        Optional[bool],
        Option("--version", callback=print_version_callback, is_eager=True),
    ] = None,
):
    """Tool to convert BrainGlobe-formatted atlases into Pinpoint V-compatible atlases."""


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

            # Extract first atlas to do common processing with.
            first_atlas = next(group_iterator)

            # Skip atlases with no root.
            if first_atlas.hierarchy.root is None:
                print(f"Atlas {group_name} is missing root! Skipping...")
                continue

            # Begin building resolution list.
            resolutions = [first_atlas.metadata["resolution"][0]]

            # Build LUTs
            structure_lut = build_structure_lut(first_atlas)
            color_lut = build_color_lut(structure_lut)

            # Save color LUT.
            save_color_lut(color_lut, atlas_path)

            # Save meshes.
            save_meshes(first_atlas, atlas_path)

            # Compress annotations.
            save_annotation(first_atlas, atlas_path)

            # Extract root ID.
            root_id = int(
                searchsorted(
                    get_sorted_structure_ids(first_atlas), first_atlas.hierarchy.root
                )
            )

            # Compress annotations and track resolutions.
            for atlas in group_iterator:
                save_annotation(atlas, atlas_path)
                resolutions.append(atlas.metadata["resolution"][0])
                progress.advance(task)

            # Save atlas metadata.
            metadata = PinpointAtlasMetadata(
                name=group_name,
                converter_version=get_converter_version(),
                resolutions=resolutions,
                root_id=root_id,
                structures=structure_lut,
                default_reference_coordinate=build_default_reference_coordinate(
                    first_atlas
                ),
            )
            with open(prepare_path(atlas_path / "atlas.json"), "w") as f:
                f.write(metadata.model_dump_json())

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


if __name__ == "__main__":
    app()
