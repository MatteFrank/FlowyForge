"""Model plugin namespace."""

from flowyforge.model_plugins.foundation_encoder import EventClassificationModel, FoundationEncoder
from flowyforge.model_plugins.perceiver import PerceiverConfig, PerceiverEventEncoder

__all__ = [
    "EventClassificationModel",
    "PerceiverConfig",
    "PerceiverEventEncoder",
    "FoundationEncoder",
]

