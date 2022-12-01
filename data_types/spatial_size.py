"""Spatial size."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SpatialSize:
    """Spatial size."""

    width: int
    height: int
    depth: int = 1
