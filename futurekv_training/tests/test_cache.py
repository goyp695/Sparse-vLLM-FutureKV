import torch

from futurekv_training.cache import build_futurekv_training_targets


def test_futurekv_targets_match_runtime_feature_width():
    attentions = torch.softmax(torch.randn(1, 4, 10, 10), dim=-1)
    key = torch.randn(1, 2, 10, 8)
    value = torch.randn(1, 2, 10, 8)

    batch = build_futurekv_training_targets(
        key,
        value,
        attentions,
        step_drop=2,
    )

    assert batch.key.shape == (1, 2, 8, 8)
    assert batch.value.shape == (1, 2, 8, 8)
    assert batch.attn_info.shape == (1, 2, 8, 16)
    assert batch.drop_scores.shape == (1, 2, 8)
