"""Data models and validation methods."""

from collections import defaultdict
from typing import Annotated, Optional

from pydantic import AfterValidator, BaseModel, Field

# Remapped structure ID (should be in the range of an unsigned short)
StructureId = Annotated[int, Field(gt=0, lt=1 << 16)]

# Unsigned byte integer.
UInt8 = Annotated[int, Field(ge=0, le=255)]

# Structure LUT which will start with None to indicate "empty" space.
type StructureLut = tuple[AtlasStructure, ...]


class AtlasStructure(BaseModel, frozen=True):
    """Structure description.

    Attributes:
        name: Full name of the structure.
        acronym: Acronym of the structure.
        parent_id: Parent of this structure if it has one (i.e. root would not have one).
        children_ids: Set of child structure IDs (leaf structures would be empty).
        color: RGB color of the structure as unsigned bytes.
    """

    name: Annotated[str, Field(min_length=1)]
    acronym: Annotated[str, Field(min_length=1)]
    parent_id: Optional[StructureId]
    children_ids: frozenset[StructureId]
    color: tuple[UInt8, UInt8, UInt8]


def ensure_sorted_and_unique(value: tuple[float, ...]) -> tuple[float, ...]:
    """Ensures the tuple is sorted and has no duplicates.

    Args:
        value: A tuple of floats to verify.

    Returns:
        Sorted tuple.

    Raises:
        ValueError: If the tuple has duplicates.
    """
    if len(set(value)) != len(value):
        raise ValueError("List must have unique values!")
    return tuple(sorted(value))


def ensure_starts_with_empty_structure_and_unique(
    value: StructureLut,
) -> StructureLut:
    """Ensures the structure LUT is sorted and has no duplicates.

    Args:
        value: A structure LUT to verify.

    Returns:
        Validated structure LUT.

    Raises:
        ValueError: If the LUT is empty, does not contain exactly 1 empty structure only at the start, or has duplicates.
    """
    if len(value) == 0:
        raise ValueError("LUT must have values!")
    elif value[0].name != "empty":
        raise ValueError("LUT must start with empty structure!")
    elif len(set(value)) != len(value):
        positions = defaultdict(list)
        for index, structure in enumerate(value):
            positions[structure].append(index)
        duplicate_positions = {
            structure: indices
            for structure, indices in positions.items()
            if len(indices) > 1
        }
        raise ValueError(
            f"LUT must have unique values! Duplicates: {duplicate_positions}"
        )
    return value


class PinpointAtlasMetadata(BaseModel, frozen=True):
    """Atlas description and metadata.

    Attributes:
        name: Name of the atlas by specimen.
        converter_version: Version number of the converter used to build this atlas.
        resolutions: Supported resolutions in sorted order.
        root_id: ID of the root structure.
        structures: Ordered list of structures in the atlas where the index is the structure ID. ID 0 is empty space.
    """

    name: Annotated[str, Field(min_length=1)]
    converter_version: str
    resolutions: Annotated[
        tuple[float, ...], Field(min_length=1), AfterValidator(ensure_sorted_and_unique)
    ]
    root_id: StructureId
    structures: Annotated[
        StructureLut,
        Field(min_length=1),
        AfterValidator(ensure_starts_with_empty_structure_and_unique),
    ]
