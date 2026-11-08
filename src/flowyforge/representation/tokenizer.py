"""Typed object projections with True = valid masks throughout the pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import nn

from flowyforge.core.typing import EventBatch


@dataclass(slots=True)
class TokenBatch:
    tokens: torch.Tensor
    mask: torch.Tensor
    group_ids: torch.Tensor
    # Collection -> half-open interval in the concatenated sequence; object
    # order within each interval is exactly the reader's sorted/truncated order.
    group_slices: dict[str, tuple[int, int]] = field(default_factory=dict)


class ObjectTokenizer(nn.Module):
    """Project collections independently and add learned type embeddings.

    Configured order defines stable group IDs. Missing groups are permitted;
    unknown groups and mismatched feature widths are errors. Masked values are
    removed before projection, including NaN/inf in padding.
    """

    def __init__(self, collection_dims: Mapping[str, int], embedding_dim: int = 64,
                 *, mask_indicators: bool = False) -> None:
        super().__init__()
        if not collection_dims or any(type(dim) is not int or dim < 1 for dim in collection_dims.values()):
            raise ValueError("collection_dims must contain positive integer feature dimensions.")
        if type(embedding_dim) is not int or embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer.")
        if any(not name or "." in name for name in collection_dims):
            raise ValueError("Collection names must be nonempty and contain no dots.")
        self.collection_dims = dict(collection_dims)
        self.embedding_dim = embedding_dim
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, embedding_dim) for name, dim in collection_dims.items()
        })
        self.type_embeddings = nn.Embedding(len(collection_dims), embedding_dim)
        self.mask_indicators = mask_indicators
        self.mask_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.empty(dim, embedding_dim)) for name, dim in collection_dims.items()
        } if mask_indicators else {})
        for parameter in self.mask_embeddings.values():
            nn.init.normal_(parameter, std=0.02)

    def forward(self, batch: EventBatch) -> TokenBatch:
        if not isinstance(batch, EventBatch):
            raise TypeError("ObjectTokenizer expects a structured EventBatch.")
        if unknown := set(batch.objects) - self.collection_dims.keys():
            raise ValueError(f"Unconfigured object groups: {sorted(unknown)}")
        tokens, masks, group_ids = [], [], []
        group_slices, offset = {}, 0
        batch_size = None
        for group_id, (name, dim) in enumerate(self.collection_dims.items()):
            if name not in batch.objects:
                continue
            group = batch.objects[name]
            features, mask = group.features, group.mask
            if not isinstance(features, torch.Tensor) or features.ndim != 3 or features.shape[-1] != dim:
                raise ValueError(f"{name}: expected features [B, N, {dim}].")
            if not features.is_floating_point():
                raise ValueError(f"{name}: features must be floating point.")
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != features.shape[:2]:
                raise ValueError(f"{name}: expected a boolean mask [B, N].")
            if features.device != self.type_embeddings.weight.device or mask.device != features.device:
                raise ValueError(f"{name}: features, masks and tokenizer must share a device.")
            if batch_size is not None and features.shape[0] != batch_size:
                raise ValueError("All object groups must share the same batch size.")
            batch_size = features.shape[0]
            hidden = group.feature_mask
            if hidden is not None:
                if hidden.dtype != torch.bool or hidden.shape != features.shape or hidden.device != features.device:
                    raise ValueError(f"{name}: feature_mask must be boolean [B, N, F] on the feature device.")
                if not self.mask_indicators:
                    raise ValueError("Feature masking requires a tokenizer with mask_indicators=True.")
                if (hidden & ~mask.unsqueeze(-1)).any():
                    raise ValueError(f"{name}: padding cannot be feature-masked.")
                features = features.masked_fill(hidden, 0)
            if not torch.isfinite(features[mask]).all():
                raise ValueError(f"{name}: non-finite features in valid objects.")
            safe = features.masked_fill(~mask.unsqueeze(-1), 0)
            projected = self.projections[name](safe) + self.type_embeddings.weight[group_id]
            if hidden is not None:
                projected = projected + hidden.to(projected.dtype) @ self.mask_embeddings[name]
            group_slices[name] = (offset, offset + features.shape[1])
            offset += features.shape[1]
            tokens.append(projected.masked_fill(~mask.unsqueeze(-1), 0))
            masks.append(mask)
            group_ids.append(torch.full(mask.shape, group_id, dtype=torch.long, device=mask.device))
        if not tokens:
            batch_size = batch.metadata.get("batch_size")
            if type(batch_size) is not int or batch_size < 1:
                raise ValueError("An EventBatch without groups needs positive metadata['batch_size'].")
            weights = self.type_embeddings.weight
            return TokenBatch(weights.new_zeros((batch_size, 0, self.embedding_dim)),
                              torch.zeros((batch_size, 0), dtype=torch.bool, device=weights.device),
                              torch.empty((batch_size, 0), dtype=torch.long, device=weights.device))
        return TokenBatch(torch.cat(tokens, dim=1), torch.cat(masks, dim=1), torch.cat(group_ids, dim=1), group_slices)

    def tokenize(self, batch: EventBatch) -> TokenBatch:
        return self(batch)


__all__ = ["TokenBatch", "ObjectTokenizer"]

