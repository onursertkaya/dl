"""VGGNet implementation.
https://arxiv.org/abs/1409.1556
"""
from dataclasses import dataclass
from typing import List, final

from interfaces.model import ModelBackbone
from zoo.blocks.vgg import VGGNetBlock, VGGNetPoolBlock


@final
class VGGNet(ModelBackbone):
    """VGGNet encoder."""

    @dataclass(frozen=True)
    class LayerConfig:
        """VGGNet configuration."""

        layer_counts: List[int]

    CONFIGS = {
        "A": LayerConfig(
            layer_counts=[1, 1, 2, 2, 2],
        ),
        "B": LayerConfig(
            layer_counts=[1, 1, 2, 2, 2],
        ),
        "C": LayerConfig(
            layer_counts=[2, 2, 3, 3, 3],
        ),
        "D": LayerConfig(
            layer_counts=[2, 2, 3, 3, 3],
        ),
        "E": LayerConfig(
            layer_counts=[2, 2, 4, 4, 4],
        ),
    }

    def __init__(self, arch: str):
        """Initialize encoder."""
        self._validate(arch)
        super().__init__()
        layer_counts = type(self).CONFIGS[arch].layer_counts

        self._layers = []

        for layer_type_idx, layer_count in enumerate(layer_counts, start=1):
            for _ in range(layer_count):
                self._layers.append(VGGNetBlock(layer_type_idx))
            self._layers.append(VGGNetPoolBlock(layer_type_idx))

    def call(self, layer_in, training):
        """Run encoder."""
        intermediate = layer_in
        for layer in self._layers:
            intermediate = layer(intermediate, training)
        return intermediate

    def _validate(self, arch: str):
        assert arch in type(self).CONFIGS, f"Unknown VGGNet architecture: {arch}"
