"""Data models and validation methods."""

from collections import defaultdict
from typing import Annotated, Any, Optional, override

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

# Remapped structure ID (should be in the range of an unsigned short)
StructureId = Annotated[int, Field(ge=0, lt=1 << 16)]

# Unsigned byte integer.
UInt8 = Annotated[int, AfterValidator(lambda value: max(0, min(255, value)))]

# Structure LUT is an ID ordered list of structures.
type StructureLut = tuple[AtlasStructure, ...]


class CamelCaseModel(BaseModel):
    """Base model that (de)serializes fields using camelCase JSON keys."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    @override
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("by_alias", True)
        return super().model_dump(**kwargs)

    @override
    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("by_alias", True)
        return super().model_dump_json(**kwargs)


class AtlasStructure(CamelCaseModel, frozen=True):
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


def ensure_unique_structures(
    value: StructureLut,
) -> StructureLut:
    """Ensures the structure LUT has no duplicates.

    Args:
        value: A structure LUT to verify.

    Returns:
        Validated structure LUT.

    Raises:
        ValueError: If the LUT has duplicates.
    """
    if len(set(value)) != len(value):
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


class PinpointAtlasMetadata(CamelCaseModel, frozen=True):
    """Atlas description and metadata.

    Attributes:
        name: Name of the atlas by specimen.
        converter_version: Version number of the converter used to build this atlas.
        resolutions: Supported resolutions in sorted order.
        root_id: ID of the root structure.
        structures: Ordered list of structures in the atlas where the index is the structure ID. ID 0 is empty space.
    """

    name: Annotated[str, Field(min_length=1)]
    converter_version: Annotated[str, Field(min_length=1)]
    resolutions: Annotated[
        tuple[float, ...], Field(min_length=1), AfterValidator(ensure_sorted_and_unique)
    ]
    root_id: StructureId
    structures: Annotated[
        StructureLut,
        Field(min_length=1),
        AfterValidator(ensure_unique_structures),
    ]
