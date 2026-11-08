"""Feature masking and balanced Huber reconstruction, without process labels."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from flowyforge.core.typing import EventBatch, ObjectGroup
from flowyforge.task_plugins.base import TaskPlugin


@dataclass(slots=True)
class MaskedBatch:
    inputs: EventBatch
    targets: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]


class MaskedModelingTask(TaskPlugin):
    def __init__(self, feature_names=None, *, targets=None, probability=0.15, seed=42,
                 huber_delta=1.0, collection_weights=None, feature_weights=None):
        if feature_names is None:
            from flowyforge.data_plugins.collide_v2.object_dataset import default_collection_specs
            feature_names = {s.name: list(s.feature_names) for s in default_collection_specs()}
        self.feature_names = {name: list(names) for name, names in feature_names.items()}
        self.targets = targets if targets is not None else {
            name: [f for f in names if f == "log1p_PT" or (name == "JetPuppiAK4" and f == "log1p_Mass")]
            for name, names in self.feature_names.items() if "log1p_PT" in names}
        if not isinstance(self.targets, dict) or not self.targets:
            raise ValueError("Masking requires at least one target collection with PT or jet Mass.")
        for name, names in self.targets.items():
            allowed = {"log1p_PT", "log1p_Mass"} if name == "JetPuppiAK4" else {"log1p_PT"}
            if (name not in self.feature_names or not isinstance(names, list) or not names
                    or len(set(names)) != len(names) or set(names) - allowed
                    or set(names) - set(self.feature_names[name])):
                raise ValueError(f"Invalid masking targets for {name}: {names}")
        if isinstance(probability, bool) or not math.isfinite(probability) or not 0 < probability <= 1:
            raise ValueError("Mask probability must be in (0, 1].")
        if isinstance(huber_delta, bool) or not math.isfinite(huber_delta) or huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive.")
        if type(seed) is not int or seed < 0:
            raise ValueError("Mask seed must be a nonnegative integer.")
        self.probability, self.seed, self.huber_delta = probability, seed, huber_delta
        self.collection_weights = dict(collection_weights or {})
        self.feature_weights = dict(feature_weights or {})
        keys = {self.key(name, f) for name, names in self.targets.items() for f in names}
        for weights, allowed in ((self.collection_weights, set(self.targets)), (self.feature_weights, keys)):
            if set(weights) - allowed:
                raise ValueError(f"Unknown masking weight keys: {sorted(set(weights) - allowed)}")
            if any(isinstance(w, bool) or not math.isfinite(w) or w <= 0 for w in weights.values()):
                raise ValueError("Masking weights must be finite and positive.")

    @property
    def name(self) -> str:
        return "masked_modeling"

    @staticmethod
    def key(collection, feature):
        return f"{collection}/{feature}"

    def eligible(self, batch):
        """Original values and validity in target order; never fed to the model."""
        targets, masks = {}, {}
        for name, names in self.targets.items():
            if name not in batch.objects:
                continue
            group = batch.objects[name]
            if group.metadata.get("feature_names") != self.feature_names[name]:
                raise ValueError(f"{name}: feature order differs from masking configuration.")
            validity = group.feature_valid
            if validity is None or validity.dtype != torch.bool or validity.shape != group.features.shape:
                raise ValueError(f"{name}: original feature_valid boolean [B, N, F] is required for masking.")
            if group.feature_mask is not None and group.feature_mask.any():
                raise ValueError("Expected an uncorrupted batch; nested masking is unsupported.")
            indices = [self.feature_names[name].index(feature) for feature in names]
            values = group.features[..., indices]
            targets[name] = values
            masks[name] = validity[..., indices] & group.mask[..., None] & torch.isfinite(values)
        return targets, masks

    def prepare_batch(self, batch: EventBatch, *, epoch=0, validation=False) -> MaskedBatch:
        targets, eligible = self.eligible(batch)
        events = batch.metadata.get("events", [batch.metadata])
        if len(events) != batch.metadata.get("batch_size") or any(not e.get("event_id") for e in events):
            raise ValueError("Masking requires stable event_id metadata for every event.")
        groups, masks = {}, {}
        for name, group in batch.objects.items():
            hidden = torch.zeros_like(group.features, dtype=torch.bool)
            if name in eligible:
                draws = []
                for event in events:
                    # No view, batch position, rank or worker ID enters this seed.
                    payload = f"mask-v1|{self.seed}|{'validation' if validation else 'train'}|{0 if validation else epoch}|{event['event_id']}|{name}"
                    seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63)
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    draws.append(torch.rand(eligible[name].shape[1:], generator=generator))
                selected = torch.stack(draws).to(group.features.device) < self.probability
                masks[name] = selected & eligible[name]
                for index, feature in enumerate(self.targets[name]):
                    hidden[..., self.feature_names[name].index(feature)] = masks[name][..., index]
            groups[name] = ObjectGroup(name, group.features.masked_fill(hidden, 0), group.mask,
                                       group.metadata, group.feature_valid, hidden)
        # Targets/labels cannot reach the encoder or decoder through this object.
        return MaskedBatch(EventBatch(groups, targets=None, metadata=batch.metadata), targets, masks)

    def loss_statistics(self, outputs, batch: MaskedBatch):
        """Sums and counts per feature, allowing batch-order-independent evaluation."""
        statistics = {}
        for name, names in self.targets.items():
            if name not in batch.targets:
                continue
            if name not in outputs or outputs[name].shape != batch.targets[name].shape:
                raise ValueError(f"{name}: decoder output shape differs from reconstruction targets.")
            for index, feature in enumerate(names):
                mask = batch.masks[name][..., index]
                count = int(mask.sum().item())
                if count:
                    # Select before computing loss: NaN in an excluded target is harmless.
                    errors = F.huber_loss(outputs[name][..., index][mask], batch.targets[name][..., index][mask],
                                          reduction="sum", delta=self.huber_delta)
                    statistics[self.key(name, feature)] = (errors, count)
        return statistics

    def combine(self, statistics):
        collections, weights = [], []
        for name, features in self.targets.items():
            means, feature_weights = [], []
            for feature in features:
                key = self.key(name, feature)
                if key in statistics and statistics[key][1]:
                    total, count = statistics[key]
                    weight = self.feature_weights.get(key, 1.0)
                    means.append(total / count * weight)
                    feature_weights.append(weight)
            if means:
                weight = self.collection_weights.get(name, 1.0)
                collections.append(sum(means) / sum(feature_weights) * weight)
                weights.append(weight)
        return sum(collections) / sum(weights) if collections else None

    def compute_loss(self, outputs, batch: MaskedBatch):
        """None means no masked targets: skip backward AND optimizer.step()."""
        return self.combine(self.loss_statistics(outputs, batch))

    def compute_metrics(self, outputs, batch):
        if not isinstance(batch, MaskedBatch):
            return {}
        loss = self.compute_loss(outputs, batch)
        return {"masked_huber": float(loss.detach())} if loss is not None else {}


__all__ = ["MaskedBatch", "MaskedModelingTask"]
