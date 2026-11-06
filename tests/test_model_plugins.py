import torch

from flowyforge.core.typing import EventBatch, ObjectGroup
from flowyforge.model_plugins import EventClassificationModel, FoundationEncoder
from flowyforge.model_plugins.registry import MODEL_REGISTRY


def test_only_perceiver_encoder_is_registered_and_classifier_runs():
    assert MODEL_REGISTRY.list() == ["foundation_tiny"]
    encoder = FoundationEncoder()
    batch = EventBatch(objects={"PUPPIPart": ObjectGroup(
        "PUPPIPart", torch.zeros(2, 3, 6), torch.ones(2, 3, dtype=torch.bool),
    )})
    assert encoder.name == "foundation_tiny"
    assert encoder.get_embeddings(batch).shape == (2, 64)
    assert encoder(batch).shape == (2, 64)
    model = EventClassificationModel(encoder, num_classes=2)
    assert model.encoder is encoder
    assert model(batch).shape == (2, 2)
