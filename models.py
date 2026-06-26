"""Data models and validation methods."""

from typing import Annotated, Any, Optional, override

from pydantic import AfterValidator, BaseModel, Field

# Remapped structure ID (should be in the range of an unsigned short)
StructureId = Annotated[int, Field(gt=0, lt=1 << 16)]

# Unsigned byte integer.
UInt8 = Annotated[int, Field(ge=0, le=255)]

# Structure LUT which will start with None to indicate "empty" space.
type StructureLut = list[AtlasStructure]


class AtlasStructure(BaseModel):
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
    children_ids: set[StructureId]
    color: Annotated[list[UInt8], Field(min_length=3, max_length=3)]

    def __hash__(self) -> int:
        return hash(self.name)

    @override
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AtlasStructure):
            return self.name == other.name
        return False


def ensure_sorted_and_unique(value: list[float]) -> list[float]:
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


def ensure_starts_with_empty_structure_and_unique(
    value: StructureLut,
) -> StructureLut:
    """Ensures the list is sorted and has no duplicates.

    Args:
        value: A list of AtlasStructures to verify.

    Returns:
        Validated list.

    Raises:
        ValueError: If the list is empty, does not contain exactly 1 None only at the start, or has duplicates.
    """
    if len(value) == 0:
        raise ValueError("LUT must have values!")
    elif value[0].name != "empty":
        raise ValueError("LUT must start with empty structure!")
    elif len(set(value)) != len(value):
        raise ValueError("LUT must have unique values!")
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
        list[float], Field(min_length=1), AfterValidator(ensure_sorted_and_unique)
    ]
    root_id: StructureId
    structures: Annotated[
        StructureLut,
        Field(min_length=1),
        AfterValidator(ensure_starts_with_empty_structure_and_unique),
    ]
