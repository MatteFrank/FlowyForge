"""Model plugin namespace."""

from flowyforge.model_plugins.foundation_encoder import EventClassificationModel, FoundationEncoder
from flowyforge.model_plugins.perceiver import PerceiverConfig, PerceiverEventEncoder
from flowyforge.model_plugins.graph_transformer import GraphTransformerModel
from flowyforge.model_plugins.mlp import MLPModel
from flowyforge.model_plugins.tiny_transformer import TinyTransformerModel

__all__ = [
    "EventClassificationModel",
    "PerceiverConfig",
    "PerceiverEventEncoder",
    "FoundationEncoder",
    "GraphTransformerModel",
    "MLPModel",
    "TinyTransformerModel",
]

