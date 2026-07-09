"""Annotation compression operations."""

from functools import lru_cache, partial
from math import ceil, sqrt
from multiprocessing.pool import Pool
from pathlib import Path

from brainglobe_atlasapi import BrainGlobeAtlas
from numpy import array, dtype, ndarray, searchsorted, uint16, where
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, track
from trimesh import load_mesh
from zarr import create_array
from zarr.codecs import BloscCodec, BloscShuffle

from atlas_manager import prepare_path
from models import AtlasStructure, StructureLut, UInt8

type Annotation = ndarray[tuple[int, int, int], dtype[uint16]]

"""Remappings."""


@lru_cache(1)
def get_sorted_structure_ids(atlas: BrainGlobeAtlas) -> list[int]:
    """Return all structure IDs in sorted order.

    The last call is cached.

    Args:
        atlas: The atlas to extract IDs from.

    Raises:
        ValueError: If IDs exceed 16-bit value.
    """
    sorted_keys = sorted(atlas.structures.keys())

    if len(sorted_keys) - 1 >= (1 << 16):
        raise ValueError(
            f"Atlas structure IDs exceeds 16-bit data limit. We assume atlases with under {1 << 16} IDs with an additional slot for the empty region."
        )

    return sorted_keys


def build_remapped_annotation(
    atlas: BrainGlobeAtlas,
) -> Annotation:
    """Returns remap annotation values to the sorted structure IDs order.

    Empty space (encoded as 0 in the annotation) is remapped to max uint16.

    Args:
        atlas: Brain Globe atlas to remap the annotation of.
    """
    flat_atlas = atlas.annotation.ravel()

    # Find the next unused ID value.
    ids = get_sorted_structure_ids(atlas)
    search_set = set(ids)
    unused = 0
    while unused in search_set:
        unused += 1

    # Replace empty if needed.
    if unused != 0:
        flat_atlas[flat_atlas == 0] = unused

    # Remap annotation. Values not found in `ids` (e.g. relabeled empty space)
    # are mapped to max uint16.
    ids_array = array(ids)
    codes = searchsorted(ids_array, flat_atlas).clip(max=len(ids_array) - 1)
    found = ids_array[codes] == flat_atlas
    remapped_flat = where(found, codes, 65535).astype(uint16)
    return remapped_flat.reshape(atlas.shape)


"""LUT Builders."""


def build_structure_lut(atlas: BrainGlobeAtlas) -> StructureLut:
    """Returns the Structure LUT for an atlas.

    Args:
        atlas: Brain Globe atlas to build the Structure LUT for.
    """
    # Initialize the LUT.
    lut: list[AtlasStructure] = []

    # Get all structure IDs and skip the 0-index one.
    ids = get_sorted_structure_ids(atlas)
    for structure_id in track(
        ids, description="Building structure LUT...", transient=True
    ):
        # Get the structure data.
        structure_data = atlas.structures[structure_id]
        # pyrefly: ignore [bad-argument-type]
        hierarchy_node = atlas.hierarchy.get_node(structure_id)

        # Stop if the node is missing.
        if hierarchy_node is None:
            raise ValueError(f"Structure {structure_id} not found in hierarchy.")

        # Extract parent and children.
        parent_og_id = hierarchy_node.predecessor(atlas.hierarchy.identifier)
        children_og_ids = hierarchy_node.successors(atlas.hierarchy.identifier)

        # Build structure (convert parent and children).
        lut.append(
            AtlasStructure(
                name=structure_data["name"],
                acronym=structure_data["acronym"],
                parent_id=None
                if parent_og_id is None
                else searchsorted(ids, parent_og_id),
                children_ids=set(searchsorted(ids, children_og_ids)),
                color=structure_data["rgb_triplet"],
            )
        )

    return tuple(lut)


def build_color_lut(structure_lut: StructureLut) -> list[UInt8]:
    """Returns color LUT based on a structure LUT.

    Args:
        structure_lut: Structure LUT to build the color LUT from.
    """
    lut = []
    for structure in track(
        structure_lut, description="Building color LUT...", transient=True
    ):
        lut.extend([*structure.color, 255])

    return lut


"""File I/O."""


def save_annotation(atlas: BrainGlobeAtlas, atlas_path: Path):
    """Zarr compress an atlas's annotation volume and write it to disk.

    Args:
        atlas: BrainGlobe atlas to compress and save the annotation volume of.
        atlas_path: Output directory for this atlas.
    """
    chunk_width = ceil(sqrt(1_000_000 / 4 / atlas.shape[1]))
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        progress.add_task(f"Compressing {atlas.atlas_name} annotation...", total=None)
        annotation_zarr = create_array(
            store=prepare_path(atlas_path / f"{atlas.metadata['resolution'][0]}.zarr"),
            shape=atlas.shape,
            chunks=(chunk_width, atlas.shape[1], chunk_width),
            shards=(chunk_width * 3, atlas.shape[1], chunk_width * 3),
            dtype=uint16,
            compressors=BloscCodec(shuffle=BloscShuffle.bitshuffle),
            overwrite=True,
        )
        annotation_zarr[:] = build_remapped_annotation(atlas)


def save_color_lut(lut: list[int], atlas_path: Path):
    """Write color LUT as a binary array to disk.

    Args:
        lut: Color LUT to write to disk. All values must be unsigned bytes.
        atlas_path: Output directory for this atlas.
    """
    with open(prepare_path(atlas_path / "lut.bin"), "wb") as f:
        f.write(bytes(lut))


def _convert_mesh(item: tuple[int, str], atlas_path: Path):
    # Extract mesh info.
    compacted_id, mesh_path = item

    # Skip missing meshes.
    if not Path(mesh_path).is_file():
        return

    # Load.
    mesh = load_mesh(mesh_path)

    # Apply simplification and cleanup.
    mesh = mesh.simplify_quadric_decimation(percent=0.9)
    mesh.apply_scale(0.001)
    mesh.process()

    # Export as GLB.
    mesh.export(prepare_path(atlas_path / "meshes" / f"{compacted_id}.glb"))


def save_meshes(atlas: BrainGlobeAtlas, atlas_path: Path):
    """Write atlases meshes to disk as GLB with marching cube decimation.

    Args:
        atlas: BrainGlobe atlas to convert.
        atlas_path: Output directory for this atlas.
    """
    items = list(
        enumerate(
            (
                str(atlas.meshfile_from_structure(structure))
                for structure in get_sorted_structure_ids(atlas)[1:]
            ),
            start=1,
        )
    )
    convert = partial(_convert_mesh, atlas_path=atlas_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        progress.add_task("Converting Meshes...", total=None)
        with Pool() as pool:
            pool.map(convert, items, chunksize=4)
