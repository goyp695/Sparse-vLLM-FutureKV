"""Future-attention oracle and runtime-compatible judge feature construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FutureKVTrainingTargets:
    key: torch.Tensor
    value: torch.Tensor
    attn_info: torch.Tensor
    drop_scores: torch.Tensor


def build_futurekv_training_targets(
    key: torch.Tensor,
    value: torch.Tensor,
    attentions: torch.Tensor,
    *,
    step_drop: int,
) -> FutureKVTrainingTargets:
    """Build a training decision at ``sequence_length - step_drop``.

    ``drop_scores`` are negative future-attention importance, matching inference:
    the runtime negates judge output and keeps the largest resulting scores.
    """
    if key.ndim != 4 or value.shape != key.shape:
        raise ValueError("key and value must have shape [batch, kv_heads, length, head_dim].")
    if attentions.ndim != 4:
        raise ValueError("attentions must have shape [batch, query_heads, query, key].")
    length = int(key.shape[-2])
    step_drop = int(step_drop)
    if step_drop <= 0 or length <= step_drop:
        raise ValueError(
            f"step_drop must be in [1, sequence_length-1], got {step_drop} for {length}."
        )
    kv_heads = int(key.shape[1])
    query_heads = int(attentions.shape[1])
    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count.")
    groups = query_heads // kv_heads
    grouped = attentions.reshape(
        int(attentions.shape[0]), kv_heads, groups, int(attentions.shape[2]), length
    )
    recent = grouped[..., -step_drop:, :]
    total = grouped.sum(dim=-2).transpose(-1, -2)

    def window_sum(window: int) -> torch.Tensor:
        return grouped[..., -min(window, grouped.shape[-2]):, :].sum(dim=-2).transpose(-1, -2)

    features = torch.cat(
        [
            recent.max(dim=-2).values.transpose(-1, -2),
            recent.min(dim=-2).values.transpose(-1, -2),
            total,
            window_sum(16),
            window_sum(8),
            window_sum(32),
            total,
            total,
        ],
        dim=-1,
    )
    old_end = length - step_drop
    future_importance = recent.sum(dim=-2).sum(dim=2)
    return FutureKVTrainingTargets(
        key=key[:, :, :old_end].detach(),
        value=value[:, :, :old_end].detach(),
        attn_info=features[:, :, :old_end].detach(),
        drop_scores=(-future_importance[:, :, :old_end]).detach(),
    )


def pairwise_rank_loss(
    predicted_drop_scores: torch.Tensor,
    oracle_drop_scores: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    if predicted_drop_scores.shape != oracle_drop_scores.shape:
        raise ValueError("Predicted and oracle scores must have identical shapes.")
    predicted_diff = predicted_drop_scores.unsqueeze(-1) - predicted_drop_scores.unsqueeze(-2)
    oracle_diff = oracle_drop_scores.unsqueeze(-1) - oracle_drop_scores.unsqueeze(-2)
    ordered = oracle_diff > 0
    if not bool(ordered.any()):
        return predicted_drop_scores.sum() * 0.0
    return torch.relu(float(margin) - predicted_diff[ordered]).mean()
