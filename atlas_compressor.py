"""Annotation compression operations."""

from functools import lru_cache, partial
from math import ceil, sqrt
from multiprocessing.pool import Pool
from pathlib import Path

from brainglobe_atlasapi import BrainGlobeAtlas
from numpy import dtype, ndarray, searchsorted, uint16
from pandas import Index
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, track
from trimesh import load_mesh
from zarr import create_array
from zarr.codecs import BloscCodec, BloscShuffle

from atlas_manager import get_atlas_resolution, prepare_path
from models import AtlasStructure, StructureLut, UInt8

type Annotation = ndarray[tuple[int, int, int], dtype[uint16]]

# Mesh decimation: keep this fraction of faces, capped at an absolute upper
# limit so very large source meshes (e.g. human) stay lightweight.
MESH_DECIMATION_KEEP_FRACTION = 0.05
MESH_MAX_FACES = 8000

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

    Empty space (encoded as 0 in the annotation) and any value without a
    matching structure ID are remapped to max uint16.

    Args:
        atlas: Brain Globe atlas to remap the annotation of.
    """
    flat_atlas = atlas.annotation.ravel()
    ids = get_sorted_structure_ids(atlas)

    # pyrefly: ignore [bad-argument-type]
    remapped_flat = Index(ids).get_indexer(flat_atlas).astype(uint16)
    remapped_flat[flat_atlas == 0] = (1 << 16) - 1

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
    resolution_name = "-".join(str(value) for value in get_atlas_resolution(atlas))
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        progress.add_task(f"Compressing {atlas.atlas_name} annotation...", total=None)
        annotation_zarr = create_array(
            store=prepare_path(atlas_path / f"{resolution_name}.zarr"),
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

    # Simplify: keep a small fraction of faces, capped at an absolute upper
    # limit so very large source meshes (e.g. human) stay lightweight.
    target_faces = min(
        round(len(mesh.faces) * MESH_DECIMATION_KEEP_FRACTION), MESH_MAX_FACES
    )
    mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    mesh.apply_scale(0.001)
    mesh.process()

    # Smooth shading, i.e. blend normals across adjacent faces like Blender's
    # "Shade Smooth" rather than exporting flat per-face normals.
    mesh.vertex_normals

    # Export as GLB.
    mesh.export(
        prepare_path(atlas_path / "meshes" / f"{compacted_id}.glb"),
        include_normals=True,
    )


def save_meshes(atlas: BrainGlobeAtlas, atlas_path: Path):
    """Write atlases meshes to disk as GLB with quadric decimation.

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
