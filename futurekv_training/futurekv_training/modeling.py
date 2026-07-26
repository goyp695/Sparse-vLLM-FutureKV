"""Judge architecture kept tensor-compatible with native FutureKV inference."""

from __future__ import annotations

import torch
from torch import nn


class HeadWiseLinear(nn.Module):
    def __init__(
        self,
        num_heads: int,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        std: float = 1e-2,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_heads, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_heads, out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = torch.einsum("...hi,hio->...ho", values, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output


class SelectionModel(nn.Module):
    """Predict per-token eviction scores independently for every KV head."""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = int(layer_idx)
        head_dim = int(
            getattr(
                config,
                "head_dim",
                int(config.hidden_size) // int(config.num_attention_heads),
            )
        )
        num_attention_heads = int(config.num_attention_heads)
        num_kv_heads = int(config.num_key_value_heads)
        if num_attention_heads % num_kv_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        attention_feature_dim = (num_attention_heads // num_kv_heads) * 8
        feature_dim = head_dim * 2 + attention_feature_dim
        self.fc1 = HeadWiseLinear(num_kv_heads, feature_dim, 16, std=1e-2)
        self.activation = nn.GELU()
        self.fc2 = HeadWiseLinear(num_kv_heads, 16, 1, std=1e-4)

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_info: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del token_type_ids
        features = torch.cat(
            [key.transpose(1, 2), value.transpose(1, 2), attn_info.transpose(1, 2)],
            dim=-1,
        )
        return self.fc2(self.activation(self.fc1(features)))


class _JudgeAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.judge_model = SelectionModel(config, layer_idx)


class _JudgeLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = _JudgeAttention(config, layer_idx)


class JudgeCollection(nn.Module):
    """Container whose state keys follow ``layers.N.self_attn.judge_model``."""

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [_JudgeLayer(config, index) for index in range(int(config.num_hidden_layers))]
        )

    def judge(self, layer_idx: int) -> SelectionModel:
        return self.layers[int(layer_idx)].self_attn.judge_model
