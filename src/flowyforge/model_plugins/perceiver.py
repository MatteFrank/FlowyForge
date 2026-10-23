"""A small latent Perceiver with bounded latent self-attention cost."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from flowyforge.representation.tokenizer import TokenBatch


@dataclass(frozen=True, slots=True)
class PerceiverConfig:
    embedding_dim: int = 64
    num_latents: int = 16
    num_heads: int = 4
    num_layers: int = 2
    ff_multiplier: int = 4
    dropout: float = 0.1
    output_dim: int = 64

    def __post_init__(self) -> None:
        for name in ("embedding_dim", "num_latents", "num_heads", "num_layers", "ff_multiplier", "output_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1).")


def _feed_forward(config: PerceiverConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(config.embedding_dim),
        nn.Linear(config.embedding_dim, config.embedding_dim * config.ff_multiplier),
        nn.GELU(), nn.Dropout(config.dropout),
        nn.Linear(config.embedding_dim * config.ff_multiplier, config.embedding_dim),
        nn.Dropout(config.dropout),
    )


class _LatentBlock(nn.Module):
    def __init__(self, config: PerceiverConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.attention = nn.MultiheadAttention(config.embedding_dim, config.num_heads,
                                               dropout=config.dropout, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.ff = _feed_forward(config)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(latents)
        update, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        latents = latents + self.dropout(update)
        return latents + self.ff(latents)


class PerceiverEventEncoder(nn.Module):
    def __init__(self, config: PerceiverConfig | None = None) -> None:
        super().__init__()
        self.config = config or PerceiverConfig()
        cfg = self.config
        self.latents = nn.Parameter(torch.empty(cfg.num_latents, cfg.embedding_dim))
        nn.init.normal_(self.latents, std=0.02)
        self.query_norm = nn.LayerNorm(cfg.embedding_dim)
        self.context_norm = nn.LayerNorm(cfg.embedding_dim)
        self.cross_attention = nn.MultiheadAttention(cfg.embedding_dim, cfg.num_heads,
                                                     dropout=cfg.dropout, batch_first=True)
        self.cross_dropout = nn.Dropout(cfg.dropout)
        self.cross_ff = _feed_forward(cfg)
        self.layers = nn.ModuleList([_LatentBlock(cfg) for _ in range(cfg.num_layers)])
        self.output = nn.Sequential(nn.LayerNorm(cfg.embedding_dim), nn.Linear(cfg.embedding_dim, cfg.output_dim))

    def forward(self, batch: TokenBatch) -> torch.Tensor:
        tokens, mask = batch.tokens, batch.mask
        if tokens.ndim != 3 or tokens.shape[-1] != self.config.embedding_dim:
            raise ValueError("Perceiver tokens must have shape [B, N, embedding_dim].")
        if mask.dtype != torch.bool or mask.shape != tokens.shape[:2]:
            raise ValueError("Perceiver mask must be boolean [B, N].")
        if mask.device != tokens.device or tokens.device != self.latents.device:
            raise ValueError("Tokens, mask and Perceiver must share a device.")
        if not torch.isfinite(tokens[mask]).all():
            raise ValueError("Non-finite valid object tokens.")
        tokens = tokens.masked_fill(~mask.unsqueeze(-1), 0)
        # Only empty events can attend to the zero sentinel, avoiding all-masked
        # softmax rows and NaN gradients without exposing padding as data.
        empty = ~mask.any(dim=1, keepdim=True)
        context = torch.cat((tokens, tokens.new_zeros((tokens.shape[0], 1, tokens.shape[2]))), dim=1)
        valid = torch.cat((mask, empty), dim=1)
        latents = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        context = self.context_norm(context)
        update, _ = self.cross_attention(self.query_norm(latents), context, context,
                                         key_padding_mask=~valid, need_weights=False)
        latents = latents + self.cross_dropout(update)
        latents = latents + self.cross_ff(latents)
        for layer in self.layers:
            latents = layer(latents)
        return self.output(latents.mean(dim=1))


__all__ = ["PerceiverConfig", "PerceiverEventEncoder"]
