"""Shared object foundation encoder and a minimal classification task head."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from flowyforge.core.typing import EventBatch
from flowyforge.model_plugins.base import ModelPlugin
from flowyforge.model_plugins.perceiver import PerceiverConfig, PerceiverEventEncoder
from flowyforge.representation.tokenizer import ObjectTokenizer, TokenBatch


@dataclass(slots=True)
class FoundationFeatures:
    event_embedding: torch.Tensor
    tokens: TokenBatch
    object_metadata: dict
    event_metadata: dict


class FoundationEncoder(ModelPlugin):
    def __init__(
        self,
        collection_dims: Mapping[str, int] | None = None,
        config: PerceiverConfig | None = None,
        *, mask_indicators: bool = False,
    ) -> None:
        super().__init__()
        if collection_dims is None:
            from flowyforge.data_plugins.collide_v2.object_dataset import collection_feature_dims

            collection_dims = collection_feature_dims()
        self.config = config or PerceiverConfig()
        self.embedding_dim = self.config.embedding_dim
        self.output_dim = self.config.output_dim
        self.tokenizer = ObjectTokenizer(collection_dims, self.embedding_dim, mask_indicators=mask_indicators)
        self.encoder = PerceiverEventEncoder(self.config)

    @property
    def name(self) -> str:
        return "foundation_tiny"

    def encode(self, batch: EventBatch) -> torch.Tensor:
        return self.forward_features(batch).event_embedding

    def forward_features(self, batch: EventBatch) -> FoundationFeatures:
        """Tokenize once; return the event embedding and object-aligned tokens."""
        tokens = self.tokenizer(batch)
        return FoundationFeatures(self.encoder(tokens), tokens,
                                  {name: group.metadata for name, group in batch.objects.items()}, batch.metadata)

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

    @classmethod
    def from_encoder_checkpoint(cls, path, num_classes: int, *, expected_contract=None):
        """Strictly restore a transferable encoder and initialize a new head."""
        return cls(load_encoder(path, expected_contract=expected_contract), num_classes)


class MaskedReconstructionModel(nn.Module):
    """Collection decoders see only corrupted tokens and the global embedding."""

    def __init__(self, encoder: FoundationEncoder, targets: Mapping[str, list[str]], hidden_dim: int = 64):
        super().__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(encoder.embedding_dim + encoder.output_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, len(features))) for name, features in targets.items()})

    def forward(self, corrupted: EventBatch) -> dict[str, torch.Tensor]:
        features = self.encoder.forward_features(corrupted)
        outputs = {}
        for name, decoder in self.decoders.items():
            if name in features.tokens.group_slices:
                start, stop = features.tokens.group_slices[name]
                tokens = features.tokens.tokens[:, start:stop]
                context = features.event_embedding[:, None, :].expand(-1, stop - start, -1)
                outputs[name] = decoder(torch.cat((tokens, context), dim=-1))
        return outputs


def encoder_contract(model: Mapping, data: Mapping) -> dict:
    """Versioned input semantics, independent of cache locations and batch sizes."""
    from flowyforge.data_plugins.collide_v2.object_dataset import default_collection_specs

    specs = default_collection_specs(data["collection_limits"])
    return {"version": 1, "architecture": asdict(PerceiverConfig(**{k: v for k, v in model.items() if k != "name"})),
            "mask_indicators": True,
            "reader": {key: data[key] for key in ("view", "units", "collection_limits", "eta_clip", "value_clip")},
            "collections": [{"name": s.name, "fields": list(s.fields), "optional_fields": list(s.optional_fields),
                             "feature_names": list(s.feature_names), "max_objects": s.max_objects,
                             "sort_by_pt": s.sort_by_pt} for s in specs],
            "transforms": {"PT/MET/Mass": "log1p(clip(x,0,value_clip)) after unit conversion",
                           "Eta": "clip(x,-eta_clip,eta_clip)/eta_clip", "Phi": "sin(phi),cos(phi) in radians",
                           "PID": "sign(x)*log1p(min(abs(x),value_clip))", "Charge": "clip(x,-3,3)",
                           "invalid": "missing/nonfinite -> zero; negative PT/MET/Mass invalid for masking",
                           "padding": "object mask False; feature validity and feature masking False",
                           "masking": "hidden value -> zero plus learned indicator per collection/feature"}}


def export_encoder(path, encoder: FoundationEncoder, contract: dict, provenance: dict) -> None:
    """Atomic export; the training runner owns overwrite protection."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"kind": "flowyforge.masked_encoder", "version": 1, "contract": contract,
                "provenance": provenance,
                "state_dict": {k: v.detach().cpu() for k, v in encoder.state_dict().items()}}, temporary)
    temporary.replace(path)


def load_encoder(path, *, expected_contract: Mapping | None = None) -> FoundationEncoder:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("kind") != "flowyforge.masked_encoder" or payload.get("version") != 1:
        raise ValueError("Unsupported transferable encoder checkpoint kind/version.")
    contract = payload.get("contract", {})
    try:
        canonical = encoder_contract(contract["architecture"], contract["reader"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid encoder architecture/input contract.") from exc
    if contract != canonical or (expected_contract is not None and dict(expected_contract) != contract):
        raise ValueError("Incompatible encoder contract: check view, architecture, feature order, transforms and object limits.")
    dims = {c["name"]: len(c["feature_names"]) for c in contract["collections"]}
    encoder = FoundationEncoder(dims, PerceiverConfig(**contract["architecture"]), mask_indicators=True)
    try:
        encoder.load_state_dict(payload["state_dict"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise ValueError(f"Incompatible encoder weights (strict key/shape check): {exc}") from exc
    encoder.input_contract = contract
    return encoder


__all__ = ["FoundationEncoder", "FoundationFeatures", "EventClassificationModel", "MaskedReconstructionModel",
           "encoder_contract", "export_encoder", "load_encoder"]

