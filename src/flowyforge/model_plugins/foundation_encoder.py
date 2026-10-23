"""Shared object foundation encoder and a minimal classification task head."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from flowyforge.core.typing import EventBatch
from flowyforge.model_plugins.base import ModelPlugin
from flowyforge.model_plugins.perceiver import PerceiverConfig, PerceiverEventEncoder
from flowyforge.representation.tokenizer import ObjectTokenizer


class FoundationEncoder(ModelPlugin):
    def __init__(
        self,
        collection_dims: Mapping[str, int] | None = None,
        config: PerceiverConfig | None = None,
    ) -> None:
        super().__init__()
        if collection_dims is None:
            from flowyforge.data_plugins.collide_v2.object_dataset import collection_feature_dims

            collection_dims = collection_feature_dims()
        self.config = config or PerceiverConfig()
        self.embedding_dim = self.config.embedding_dim
        self.output_dim = self.config.output_dim
        self.tokenizer = ObjectTokenizer(collection_dims, self.embedding_dim)
        self.encoder = PerceiverEventEncoder(self.config)

    @property
    def name(self) -> str:
        return "foundation_tiny"

    def encode(self, batch: EventBatch) -> torch.Tensor:
        return self.encoder(self.tokenizer(batch))

    def forward(self, batch: EventBatch, task: str | None = None) -> torch.Tensor:
        return self.encode(batch)


class EventClassificationModel(ModelPlugin):
    def __init__(self, encoder: FoundationEncoder, num_classes: int) -> None:
        super().__init__()
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("Classification requires at least two classes.")
        self.encoder = encoder
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.LayerNorm(encoder.output_dim),
            nn.Linear(encoder.output_dim, encoder.output_dim), nn.GELU(),
            nn.Linear(encoder.output_dim, num_classes),
        )

    @property
    def name(self) -> str:
        return "event_classification"

    def encode(self, batch: EventBatch) -> torch.Tensor:
        return self.encoder.encode(batch)

    def forward(self, batch: EventBatch, task: str | None = None) -> torch.Tensor:
        if task not in {None, "classification"}:
            raise ValueError(f"Unsupported task: {task}")
        return self.head(self.encode(batch))


__all__ = ["FoundationEncoder", "EventClassificationModel"]

