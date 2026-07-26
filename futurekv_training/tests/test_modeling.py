from types import SimpleNamespace

import torch

from futurekv_training.modeling import SelectionModel


def test_selection_model_matches_runtime_tensor_shape():
    config = SimpleNamespace(
        hidden_size=32,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = SelectionModel(config)
    output = model(
        torch.randn(1, 2, 7, 8),
        torch.randn(1, 2, 7, 8),
        torch.randn(1, 2, 7, 16),
    )
    assert output.shape == (1, 7, 2, 1)


def test_selection_model_backward_updates_only_judge():
    config = SimpleNamespace(
        hidden_size=32,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = SelectionModel(config)
    output = model(
        torch.randn(1, 2, 7, 8),
        torch.randn(1, 2, 7, 8),
        torch.randn(1, 2, 7, 16),
    )
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
