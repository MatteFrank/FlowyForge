import torch

from flowyforge.core.typing import EventBatch, ObjectGroup

from flowyforge.model_plugins import (
    FoundationEncoder,
    GraphTransformerModel,
    MLPModel,
    TinyTransformerModel,
)


def test_model_plugins_instantiate_and_forward() -> None:
    models = [
        MLPModel(input_dim=1),
        TinyTransformerModel(),
        GraphTransformerModel(),
        FoundationEncoder(),
    ]

    assert [model.name for model in models] == [
        "mlp",
        "tiny_transformer",
        "graph_transformer",
        "foundation_tiny",
    ]
    assert isinstance(models[0](torch.zeros(1, 1)), torch.Tensor)
    assert all(isinstance(model.get_embeddings({}), torch.Tensor) for model in models[:3])
    batch = EventBatch(objects={"PUPPIPart": ObjectGroup(
        "PUPPIPart", torch.zeros(2, 3, 6), torch.ones(2, 3, dtype=torch.bool),
    )})
    assert models[3].get_embeddings(batch).shape == (2, 64)
    assert models[3](batch).shape == (2, 64)
