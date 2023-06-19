"""Selection of models."""
from core.settings import BackboneConfig
from interfaces.model import ModelBackbone
from zoo.models.resnet import Resnet

_BACKBONES = {
    m.__name__: m
    for m in [
        Resnet,
    ]
}


def make_backbone_from_config(config: BackboneConfig) -> ModelBackbone:
    """Make a backbone from configuration."""
    backbone_type = _BACKBONES[config.name]
    return backbone_type(**config.init_kwargs)
