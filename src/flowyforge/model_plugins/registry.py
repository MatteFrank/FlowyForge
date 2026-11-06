"""Default model registry helpers."""

from __future__ import annotations

from flowyforge.core.registry import PluginRegistry
from flowyforge.model_plugins.foundation_encoder import FoundationEncoder


MODEL_REGISTRY = PluginRegistry()


def register_default_models(registry: PluginRegistry = MODEL_REGISTRY) -> PluginRegistry:
    if registry.list():
        return registry
    registry.register("foundation_tiny", FoundationEncoder)
    return registry


register_default_models()


__all__ = ["MODEL_REGISTRY", "register_default_models"]

