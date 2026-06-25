"""Default model registry helpers."""

from __future__ import annotations

from flowyforge.core.registry import PluginRegistry
from flowyforge.model_plugins.foundation_encoder import FoundationEncoder
from flowyforge.model_plugins.graph_transformer import GraphTransformerModel
from flowyforge.model_plugins.mlp import MLPModel
from flowyforge.model_plugins.tiny_transformer import TinyTransformerModel


MODEL_REGISTRY = PluginRegistry()


def register_default_models(registry: PluginRegistry = MODEL_REGISTRY) -> PluginRegistry:
    if registry.list():
        return registry
    registry.register("mlp", MLPModel)
    registry.register("tiny_transformer", TinyTransformerModel)
    registry.register("graph_transformer", GraphTransformerModel)
    registry.register("foundation_tiny", FoundationEncoder)
    return registry


register_default_models()


__all__ = ["MODEL_REGISTRY", "register_default_models"]

