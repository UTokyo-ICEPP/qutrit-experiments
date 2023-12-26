"""Recursively extract the metadata from experiment data."""
from collections.abc import Sequence
from typing import Any, Union


def get_metadata(
    metadata: dict[str, Any],
    composite_index: Union[None, int, Sequence[int]]
) -> dict[str, Any]:
    """Return the metadata from a circuit made by a possibly composite experiment."""
    if composite_index is not None:
        if isinstance(composite_index, int):
            composite_index = [composite_index]

        for idx in composite_index:
            metadata = metadata['composite_metadata'][idx]

    return metadata
