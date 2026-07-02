"""Atlas converter pipeline for Pinpoint V

Build Pinpoint V compatible atlases from BrainGlobe-style atlases.
"""

from pathlib import Path
from typing import Annotated

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
    allen_mouse_atlases,
    build_atlas_path,
    build_default_converted_atlases_path,
    build_pinpoint_atlas_metadata,
    save_pinpoint_atlas_metadata,
    save_pinpoint_atlas_metadata_schema,
)

app = Typer()


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

    # Write the metadata schema.
    save_pinpoint_atlas_metadata_schema(converted_atlases_path)

    # Build atlas group.
    first_atlas = allen_mouse_atlases()[0]
    atlas_path = build_atlas_path("allen_mouse", converted_atlases_path)

    with Progress() as progress:
        task = progress.add_task("Converting Allen CCF Mouse atlases...", total=9)

        # Compute structures and color LUTs.
        structure_lut = build_structure_lut(first_atlas)
        progress.advance(task)

        color_lut = build_color_lut(structure_lut)
        progress.advance(task)

        # Build and save atlas metadata for group.
        save_pinpoint_atlas_metadata(
            build_pinpoint_atlas_metadata(allen_mouse_atlases(), structure_lut),
            atlas_path,
        )
        progress.advance(task)

        # Build and save structure color LUT.
        save_color_lut(color_lut, atlas_path)
        progress.advance(task)

        # Convert meshes.
        save_meshes(first_atlas, atlas_path)
        progress.advance(task)

        # Compress annotations.
        for atlas in allen_mouse_atlases():
            save_annotation(atlas, atlas_path)
            progress.advance(task)


@app.command()
def brainglobe(
    converted_atlases_path: Annotated[
        Path,
        Argument(help="Output directory and root for all atlases and atlas schema."),
    ] = build_default_converted_atlases_path(),
):
    """Convert all BrainGlobe atlases."""

    # Write the metadata schema.
    save_pinpoint_atlas_metadata_schema(converted_atlases_path)

    # Iterate through all BrainGlobe atlases.


if __name__ == "__main__":
    app()
