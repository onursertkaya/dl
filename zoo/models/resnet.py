from dataclasses import dataclass
from typing import List, final

from zoo.blocks.resnet import (
    BottleneckBlock,
    InitialPoolBlock,
    InputBlock,
    ResnetBlockA,
    ResnetBlockB,
    ResnetBlockC,
    _ResnetBlock,
)
from zoo.models.base import ModelBackbone


@final
class Resnet(ModelBackbone):
    """Resnet encoder.

    This class does not include global average pooling or fully-connected layers.

    https://arxiv.org/pdf/1512.03385.pdf

    """

    @dataclass(frozen=True)
    class LayerConfig:
        layer_counts: List[int]
        block_variants: List[_ResnetBlock]

    CONFIGS = {
        18: LayerConfig(
            layer_counts=[2, 2, 2, 2],
            block_variants=[ResnetBlockA, ResnetBlockB, ResnetBlockC],
        ),
        34: LayerConfig(
            layer_counts=[3, 4, 6, 3],
            block_variants=[ResnetBlockA, ResnetBlockB, ResnetBlockC],
        ),
        50: LayerConfig(
            layer_counts=[3, 4, 6, 3],
            block_variants=[BottleneckBlock],
        ),
        101: LayerConfig(
            layer_counts=[3, 4, 23, 3],
            block_variants=[BottleneckBlock],
        ),
        152: LayerConfig(
            layer_counts=[3, 8, 36, 3],
            block_variants=[BottleneckBlock],
        ),
    }

    def __init__(self, arch: int, variant: _ResnetBlock):
        """Initialize encoder."""
        self._validate(arch, variant)
        super().__init__()
        layer_counts = type(self).CONFIGS[arch].layer_counts

        self._layers = [InputBlock(), InitialPoolBlock()]

        for layer_type_idx, layer_count in enumerate(layer_counts):
            for idx in range(layer_count):
                should_downsample = idx == 0
                if not variant is BottleneckBlock:
                    should_downsample = should_downsample and layer_type_idx != 0

                self._layers.append(
                    variant(
                        name=f"conv{(layer_type_idx+1)}_{idx}",
                        num_filters=int(2 ** (layer_type_idx + 6)),  # 64, 128, ...
                        downsample=should_downsample,
                    )
                )

    def call(self, layer_in, training):
        t = layer_in
        for layer in self._layers:
            t = layer(t, training)
        return t

    def _validate(self, arch, variant):
        assert arch in type(self).CONFIGS, f"Unknown ResNet architecture: {arch}"
        assert (
            variant in type(self).CONFIGS[arch].block_variants
        ), f"Unknown ResNet variant: {variant}"
