from __future__ import annotations

import math
import os
import re
import json
from dataclasses import dataclass
from glob import glob
from types import SimpleNamespace

import torch
from safetensors import safe_open
from torch import nn

from sparsevllm.utils.log import logger


_JUDGE_KEY_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.self_attn\.judge_model\.(?P<name>fc[12]\.(?:weight|bias))$"
)
_JUDGE_TENSOR_NAMES = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
_CHECKPOINT_SCHEMA_VERSION = 1


class HeadWiseLinear(nn.Module):
    def __init__(self, num_heads: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.num_heads = int(num_heads)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(self.num_heads, self.in_features, self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_heads, self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("...hi,hio->...ho", x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class SelectionModel(nn.Module):
    def __init__(self, config, layer_idx: int, *, local_num_key_value_heads: int | None = None):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = int(config.num_attention_heads)
        self.total_num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_heads = int(local_num_key_value_heads or self.total_num_key_value_heads)
        if self.total_num_key_value_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "local_num_key_value_heads must divide config.num_key_value_heads: "
                f"local={self.num_key_value_heads} total={self.total_num_key_value_heads}"
            )
        query_groups = self.num_attention_heads // self.total_num_key_value_heads
        attn_feature_dim = int(query_groups) * 8
        total_feature_dim = int(self.head_dim) * 2 + attn_feature_dim
        hidden_dim = 16
        self.fc1 = HeadWiseLinear(self.num_key_value_heads, total_feature_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = HeadWiseLinear(self.num_key_value_heads, hidden_dim, 1)

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_info: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del token_type_ids
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn_info = attn_info.transpose(1, 2)
        feature_vector = torch.cat([torch.cat([key, value], dim=-1), attn_info], dim=-1)
        hidden = self.activation(self.fc1(feature_vector))
        return self.fc2(hidden)


@dataclass
class FutureKVLayerRuntimeState:
    attn_acc: torch.Tensor | None = None
    attn_acc_decay: torch.Tensor | None = None
    query_cache: torch.Tensor | None = None
    query_start: int = 0
    query_length: int = 0

    def reset(self):
        self.attn_acc = None
        self.attn_acc_decay = None
        self.query_cache = None
        self.query_start = 0
        self.query_length = 0

    def reset_attention_stats(self):
        self.attn_acc = None
        self.attn_acc_decay = None

    def append_queries(self, query_states: torch.Tensor, window_size: int):
        # query_states: [Q, num_q_heads, head_dim]
        query_states = query_states.transpose(0, 1).contiguous()
        window_size = int(window_size)
        if window_size <= 0:
            if self.query_cache is None:
                self.query_cache = query_states
            else:
                self.query_cache = torch.cat([self.query_cache, query_states], dim=1)
            self.query_start = 0
            self.query_length = int(self.query_cache.shape[1])
            return

        shape = (int(query_states.shape[0]), window_size, int(query_states.shape[2]))
        if (
            self.query_cache is None
            or tuple(self.query_cache.shape) != shape
            or self.query_cache.dtype != query_states.dtype
            or self.query_cache.device != query_states.device
        ):
            self.query_cache = torch.empty(
                shape,
                dtype=query_states.dtype,
                device=query_states.device,
            )
            self.query_start = 0
            self.query_length = 0

        num_queries = int(query_states.shape[1])
        if num_queries >= window_size:
            self.query_cache.copy_(query_states[:, -window_size:, :])
            self.query_start = 0
            self.query_length = window_size
            return

        write_start = (self.query_start + self.query_length) % window_size
        first_count = min(num_queries, window_size - write_start)
        self.query_cache[:, write_start:write_start + first_count, :].copy_(
            query_states[:, :first_count, :]
        )
        if first_count < num_queries:
            self.query_cache[:, :num_queries - first_count, :].copy_(
                query_states[:, first_count:, :]
            )

        overflow = max(0, self.query_length + num_queries - window_size)
        self.query_start = (self.query_start + overflow) % window_size
        self.query_length = min(window_size, self.query_length + num_queries)

    def get_queries(self) -> torch.Tensor:
        if self.query_cache is None or self.query_length == 0:
            raise RuntimeError("FutureKV query cache is empty.")
        if self.query_length == int(self.query_cache.shape[1]) and self.query_start == 0:
            queries = self.query_cache
        elif self.query_start + self.query_length <= int(self.query_cache.shape[1]):
            queries = self.query_cache[:, self.query_start:self.query_start + self.query_length, :]
        else:
            queries = torch.cat(
                [
                    self.query_cache[:, self.query_start:, :],
                    self.query_cache[:, :self.query_start + self.query_length - self.query_cache.shape[1], :],
                ],
                dim=1,
            )
        return queries.unsqueeze(0)

    def gather_keep_indices(self, keep_indices: torch.Tensor):
        if keep_indices.dim() == 2:
            if self.attn_acc is not None:
                idx = keep_indices.view(1, keep_indices.shape[0], keep_indices.shape[1], 1)
                idx = idx.expand(self.attn_acc.shape[0], -1, -1, self.attn_acc.shape[-1])
                self.attn_acc = self.attn_acc.gather(2, idx)
            if self.attn_acc_decay is not None:
                idx = keep_indices.view(1, keep_indices.shape[0], keep_indices.shape[1], 1)
                idx = idx.expand(self.attn_acc_decay.shape[0], -1, -1, self.attn_acc_decay.shape[-1])
                self.attn_acc_decay = self.attn_acc_decay.gather(2, idx)
            return
        if self.attn_acc is not None:
            self.attn_acc = self.attn_acc.index_select(2, keep_indices)
        if self.attn_acc_decay is not None:
            self.attn_acc_decay = self.attn_acc_decay.index_select(2, keep_indices)


def _checkpoint_files(path: str | os.PathLike[str]) -> tuple[list[str], bool]:
    path = os.fspath(path)
    if os.path.isdir(path):
        canonical = os.path.join(path, "judge_model.pt")
        if os.path.isfile(canonical):
            return [canonical], False
        files = sorted(glob(os.path.join(path, "*.safetensors")))
        if files:
            return files, True
        files = sorted(glob(os.path.join(path, "*.pt")) + glob(os.path.join(path, "*.bin")))
        return files, False
    if os.path.isfile(path):
        return [path], path.endswith(".safetensors")
    return [], False


def _iter_judge_tensors(path: str):
    files, is_safetensors = _checkpoint_files(path)
    for file in files:
        if is_safetensors:
            with safe_open(file, "pt", "cpu") as f:
                for key in f.keys():
                    if "judge_model" in key:
                        yield key, f.get_tensor(key)
        else:
            state = torch.load(file, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                continue
            for key, value in state.items():
                if "judge_model" in key and torch.is_tensor(value):
                    yield key, value


def _load_checkpoint_metadata(path: str | os.PathLike[str]) -> dict:
    path = os.fspath(path)
    directory = path if os.path.isdir(path) else os.path.dirname(path)
    metadata_path = os.path.join(directory, "judge_model_meta.json")
    if not os.path.isfile(metadata_path):
        raise ValueError(
            "FutureKV checkpoint metadata is missing: "
            f"expected {metadata_path} next to judge_model.pt."
        )
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    required = {
        "schema_version",
        "base_model",
        "hidden_size",
        "intermediate_size",
        "num_layers",
        "num_kv_heads",
        "dtype",
        "training_method",
        "futurekv",
        "creation",
        "num_tensors",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"FutureKV checkpoint metadata is missing fields: {sorted(missing)}.")
    if int(metadata["schema_version"]) != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported FutureKV checkpoint schema_version={metadata['schema_version']}; "
            f"expected {_CHECKPOINT_SCHEMA_VERSION}."
        )
    return metadata


def load_futurekv_judge_state(
    path: str | os.PathLike[str],
) -> dict[int, dict[str, torch.Tensor]]:
    metadata = _load_checkpoint_metadata(path)
    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in _iter_judge_tensors(path):
        match = _JUDGE_KEY_RE.search(key)
        if match is None:
            continue
        layer_idx = int(match.group("layer"))
        by_layer.setdefault(layer_idx, {})[match.group("name")] = tensor
    expected_layers = set(range(int(metadata["num_layers"])))
    if set(by_layer) != expected_layers:
        raise ValueError(
            "FutureKV checkpoint layers do not match metadata: "
            f"found={sorted(by_layer)} expected={sorted(expected_layers)}."
        )
    tensor_count = sum(len(layer) for layer in by_layer.values())
    if tensor_count != int(metadata["num_tensors"]):
        raise ValueError(
            "FutureKV checkpoint tensor count does not match metadata: "
            f"found={tensor_count} expected={metadata['num_tensors']}."
        )
    expected_heads = int(metadata["num_kv_heads"])
    for layer_idx, layer in by_layer.items():
        missing = _JUDGE_TENSOR_NAMES.difference(layer)
        if missing:
            raise ValueError(
                f"FutureKV judge layer {layer_idx} is missing tensors: {sorted(missing)}."
            )
        if any(int(tensor.shape[0]) != expected_heads for tensor in layer.values()):
            raise ValueError(
                f"FutureKV judge layer {layer_idx} num_kv_heads does not match "
                f"metadata num_kv_heads={expected_heads}."
            )
    return by_layer


def _slice_heads(weight: torch.Tensor, *, rank: int, world_size: int, local_heads: int, total_heads: int) -> torch.Tensor:
    if int(weight.shape[0]) == local_heads:
        return weight
    if int(weight.shape[0]) != total_heads:
        raise ValueError(
            f"FutureKV judge head count mismatch: weight has {int(weight.shape[0])}, "
            f"expected local={local_heads} or total={total_heads}."
        )
    start = int(rank) * int(local_heads)
    end = start + int(local_heads)
    return weight[start:end]


def _resolve_language_config(hf_config):
    if hasattr(hf_config, "num_key_value_heads"):
        return hf_config
    for attr in ("text_config", "language_config", "llm_config"):
        sub_config = getattr(hf_config, attr, None)
        if sub_config is not None and hasattr(sub_config, "num_key_value_heads"):
            return sub_config
    return hf_config


class FutureKVJudgeBank(nn.Module):
    def __init__(
        self,
        config,
        *,
        rank: int,
        world_size: int,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.config = config
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = torch.device(device)
        self.hf_config = _resolve_language_config(config.hf_config)
        self.total_kv_heads = int(self.hf_config.num_key_value_heads)
        self.local_kv_heads = self.total_kv_heads // self.world_size
        self.num_layers = int(self.hf_config.num_hidden_layers)
        self.modules_by_layer = nn.ModuleList(
            [
                SelectionModel(self.hf_config, layer_idx, local_num_key_value_heads=self.local_kv_heads)
                for layer_idx in range(self.num_layers)
            ]
        )
        self.enabled = False
        self.source_path: str | None = None
        self.loaded_tensors = 0

        path = getattr(config, "futurekv_judge_path", None)
        if not path:
            return
        self.load_from_path(path)

    def load_from_path(self, path: str):
        state_by_layer = load_futurekv_judge_state(path)
        if not state_by_layer:
            return

        dtype = getattr(self.hf_config, "torch_dtype", getattr(self.config.hf_config, "torch_dtype", torch.float16))
        loaded_tensors = 0
        required = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
        for layer_idx, module in enumerate(self.modules_by_layer):
            layer_state = state_by_layer.get(layer_idx)
            if not layer_state:
                continue
            missing = required.difference(layer_state)
            if missing:
                raise RuntimeError(f"FutureKV judge layer {layer_idx} is missing tensors: {sorted(missing)}")

            local_state = {}
            for name, weight in layer_state.items():
                sliced = _slice_heads(
                    weight,
                    rank=self.rank,
                    world_size=self.world_size,
                    local_heads=self.local_kv_heads,
                    total_heads=self.total_kv_heads,
                )
                local_state[name] = sliced.to(device=self.device, dtype=dtype)

            incompatible = module.load_state_dict(local_state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"FutureKV judge layer {layer_idx} load mismatch: "
                    f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
                )
            module.to(device=self.device, dtype=dtype)
            module.eval()
            loaded_tensors += len(local_state)

        self.loaded_tensors = loaded_tensors
        self.enabled = loaded_tensors > 0
        self.source_path = path if self.enabled else None
        if self.enabled:
            logger.info(
                f"Loaded FutureKV judge tensors from {path}: tensors={loaded_tensors} "
                f"rank={self.rank}/{self.world_size} local_kv_heads={self.local_kv_heads}"
            )

    @torch.no_grad()
    def score(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_info: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled:
            raise RuntimeError("FutureKV judge is not loaded.")
        module = self.modules_by_layer[int(layer_idx)]
        scores = -module(key, value, attn_info)
        if scores.dim() == 4:
            scores = scores.squeeze(-1)
        if scores.dim() != 3:
            raise RuntimeError(f"Unexpected FutureKV judge score shape: {tuple(scores.shape)}")
        if scores.shape[1] != key.shape[1]:
            scores = scores.transpose(1, 2)
        return scores


def compute_futurekv_attn_info(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    step_drop: int,
    runtime_state: FutureKVLayerRuntimeState,
) -> tuple[torch.Tensor, FutureKVLayerRuntimeState]:
    if runtime_state.attn_acc is not None:
        query_states = query_states[:, :, -int(step_drop):, :]

    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads must be divisible by kv_heads, got q={q_heads} kv={kv_heads}")
    query_group_size = q_heads // kv_heads

    q = query_states.view(batch_size, kv_heads, query_group_size, q_len, head_dim)
    k = key_states.unsqueeze(2)
    attn_weights = torch.matmul(q, k.transpose(3, 4)) / math.sqrt(head_dim)

    kv_len = int(key_states.shape[-2])
    causal_mask = torch.full(
        (q_len, q_len),
        torch.finfo(attn_weights.dtype).min,
        device=attn_weights.device,
        dtype=attn_weights.dtype,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    full_mask = torch.zeros((q_len, kv_len), device=attn_weights.device, dtype=attn_weights.dtype)
    if kv_len >= q_len:
        full_mask[:, -q_len:] = causal_mask
    else:
        full_mask = causal_mask[:, :kv_len]
    attn_weights = attn_weights + full_mask.view(1, 1, 1, q_len, kv_len)
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    recent_q = min(int(step_drop), int(q_len))
    attn_weights_sum = attn_weights.sum(dim=-2).transpose(-1, -2)
    attn_weights_max = attn_weights[:, :, :, -recent_q:, :].max(dim=-2).values.transpose(-1, -2)
    attn_weights_min = attn_weights[:, :, :, -recent_q:, :].min(dim=-2).values.transpose(-1, -2)

    def window_sum(window: int) -> torch.Tensor:
        window = min(int(window), int(q_len))
        return attn_weights[:, :, :, -window:, :].sum(dim=-2).transpose(-1, -2)

    attn_weights_sum1 = window_sum(8)
    attn_weights_sum2 = window_sum(16)
    attn_weights_sum3 = window_sum(32)

    if runtime_state.attn_acc is None:
        attn_acc = attn_weights_sum
        attn_acc_decay = attn_weights_sum
    else:
        old_len = min(int(runtime_state.attn_acc.shape[2]), int(attn_weights_sum.shape[2]))
        attn_acc = torch.cat(
            [
                runtime_state.attn_acc[:, :, :old_len] + attn_weights_sum[:, :, :old_len],
                attn_weights_sum[:, :, old_len:],
            ],
            dim=2,
        )
        attn_acc_decay = torch.cat(
            [
                attn_weights_sum[:, :, :old_len] + 0.9 * runtime_state.attn_acc_decay[:, :, :old_len],
                attn_weights_sum[:, :, old_len:],
            ],
            dim=2,
        )

    runtime_state.attn_acc = attn_acc
    runtime_state.attn_acc_decay = attn_acc_decay
    length = int(step_drop)
    attn_info = torch.cat(
        [
            attn_weights_max[..., :-length, :].detach(),
            attn_weights_min[..., :-length, :].detach(),
            attn_weights_sum[..., :-length, :].detach(),
            attn_weights_sum2[..., :-length, :].detach(),
            attn_weights_sum1[..., :-length, :].detach(),
            attn_weights_sum3[..., :-length, :].detach(),
            attn_acc[..., :-length, :].detach(),
            attn_acc_decay[..., :-length, :].detach(),
        ],
        dim=-1,
    )
    return attn_info, runtime_state


def make_futurekv_test_config(
    *,
    num_layers: int = 2,
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    torch_dtype: torch.dtype = torch.float16,
):
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            head_dim=hidden_size // num_attention_heads,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            torch_dtype=torch_dtype,
        ),
        model="",
        futurekv_judge_path=None,
        futurekv_divide_length=128,
    )
