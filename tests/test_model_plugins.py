import torch

from flowyforge.model_plugins import (
    FoundationEncoder,
    GraphTransformerModel,
    MLPModel,
    TinyTransformerModel,
)


def test_model_plugins_instantiate_and_forward() -> None:
    models = [
        MLPModel(),
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
    assert all(isinstance(model.get_embeddings({}), torch.Tensor) for model in models)

