from dataclasses import dataclass


@dataclass(frozen=True)
class SpatialSize:

    width: int
    height: int
    depth: int = 1
