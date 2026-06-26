from json import dump
from typing import Annotated, Any, List, Optional, override

from brainglobe_atlasapi import BrainGlobeAtlas
from pydantic import AfterValidator, BaseModel, Field

from annotation_compressor import remapped_structure_and_color_lut
from atlas_manager import atlas_root_by_name, pinpoint_atlases_root

# Remapped structure ID (should be in the range of an unsigned short)
StructureId = Annotated[int, Field(gt=0, lt=1 << 16)]

# Unsigned byte integer.
UInt8 = Annotated[int, Field(ge=0, le=255)]


class AtlasStructure(BaseModel):
    """Structure description.

    Attributes:
        name: Full name of the structure.
        acronym: Acronym of the structure.
        parent_id: Parent of this structure if it has one (i.e. root would not have one).
        children_ids: Set of child structure IDs (leaf structures would be empty).
    """

    name: Annotated[str, Field(min_length=1)]
    acronym: Annotated[str, Field(min_length=1)]
    parent_id: Optional[StructureId]
    children_ids: set[StructureId]

    def __hash__(self) -> int:
        return hash(self.name)

    @override
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AtlasStructure):
            return self.name == other.name
        return False


def ensure_sorted_and_unique(value: List[float]) -> List[float]:
    """Ensures the list is sorted and has no duplicates.

    Args:
        value: A list of floats to verify.

    Returns:
        Sorted list.

    Raises:
        ValueError: If the list has duplicates.
    """
    if len(set(value)) != len(value):
        raise ValueError("List must have unique values!")
    return sorted(value)


def ensure_starts_with_none_and_unique(
    value: List[AtlasStructure | None],
) -> List[AtlasStructure | None]:
    """Ensures the list is sorted and has no duplicates.

    Args:
        value: A list of AtlasStructures to verify.

    Returns:
        Validated list.

    Raises:
        ValueError: If the list is empty, does not contain exactly 1 None only at the start, or has duplicates.
    """
    if len(value) == 0:
        raise ValueError("List must have values!")
    elif value[0] is not None:
        raise ValueError("List must start with None!")
    elif len(set(value)) != len(value):
        raise ValueError("List must have unique values!")
    return value


class PinpointAtlasMetadata(BaseModel):
    """Atlas description and metadata.

    Attributes:
        name: Name of the atlas by specimen.
        resolutions: Supported resolutions in sorted order.
        root_id: ID of the root structure.
        structures: Ordered list of structures in the atlas where the index is the structure ID. ID 0 is empty space.
    """

    name: Annotated[str, Field(min_length=1)]
    resolutions: Annotated[
        List[float], Field(min_length=1), AfterValidator(ensure_sorted_and_unique)
    ]
    root_id: StructureId
    structures: Annotated[
        List[AtlasStructure | None],
        Field(min_length=1),
        AfterValidator(ensure_starts_with_none_and_unique),
    ]


def pinpoint_atlas_metadata(
    atlas_group: list[BrainGlobeAtlas],
) -> PinpointAtlasMetadata:
    """Return Pinpoint Atlas metadata for a given atlas group.

    Args:
        atlas_group: Group of BrainGlobe atlases to build a Pinpoint Atlas definition for.
    Raises:
        ValueError: If the atlas group does not have a root node in the hierarchy.
    """
    # Extract first atlas for shared values.
    first_atlas = atlas_group[0]

    # Raise error of atlas doesn't have root.
    if first_atlas.hierarchy.root is None:
        raise ValueError(
            f'Root for atlas "{first_atlas.metadata["name"]}" not found in hierarchy!'
        )

    # Build output.
    return PinpointAtlasMetadata(
        name=first_atlas.metadata["name"],
        resolutions=[atlas.metadata["resolution"][0] for atlas in atlas_group],
        root_id=first_atlas.hierarchy.root,
        structures=remapped_structure_and_color_lut(first_atlas)[0],
    )


def save_pinpoint_atlas_metadata(metadata: PinpointAtlasMetadata):
    """Write Pinpoint Atlas metadata to disk.

    Args:
        metadata: Pinpoint Atlas metadata to write.
    """
    with open(atlas_root_by_name(metadata.name) / "atlas.json", "w") as f:
        f.write(metadata.model_dump_json())


def save_pinpoint_atlas_metadata_schema():
    """Write Pinpoint Atlas model schema file to output root."""
    with open(pinpoint_atlases_root() / "atlas_schema.json", "w") as f:
        dump(PinpointAtlasMetadata.model_json_schema(), f, separators=(",", ":"))
