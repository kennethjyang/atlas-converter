from pathlib import Path
from numpy import ndarray, dtype, uint32


def compress_annotation(
    annotation: ndarray[tuple[int, int, int], dtype[uint32]],
    structure_ids: list[int],
    output_directory: Path,
):
    """Compress an annotation dataset and"""
    pass
