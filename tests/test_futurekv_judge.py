import os
import math
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sparsevllm.engine.futurekv_judge import (
    FutureKVJudgeBank,
    FutureKVLayerRuntimeState,
    compute_futurekv_attn_info,
    load_futurekv_judge_state,
    make_futurekv_test_config,
)
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.sparse_controller import SparseController


def make_state_dict(num_layers=2, num_kv_heads=2, total_feature_dim=40):
    state = {}
    for layer_idx in range(num_layers):
        prefix = f"model.language_model.layers.{layer_idx}.self_attn.judge_model"
        state[f"{prefix}.fc1.weight"] = torch.randn(num_kv_heads, total_feature_dim, 16, dtype=torch.float16)
        state[f"{prefix}.fc1.bias"] = torch.randn(num_kv_heads, 16, dtype=torch.float16)
        state[f"{prefix}.fc2.weight"] = torch.randn(num_kv_heads, 16, 1, dtype=torch.float16)
        state[f"{prefix}.fc2.bias"] = torch.randn(num_kv_heads, 1, dtype=torch.float16)
    return state


def save_checkpoint(path, *, num_layers=2, num_kv_heads=2, total_feature_dim=40):
    state = make_state_dict(num_layers, num_kv_heads, total_feature_dim)
    torch.save(state, path)
    metadata = {
        "schema_version": 1,
        "base_model": "test",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "dtype": "float16",
        "training_method": "test",
        "futurekv": {
            "budget": 8,
            "window_size": 8,
            "step_drop": 2,
            "divide_length": 4,
        },
        "creation": {"seed": 0},
        "num_tensors": len(state),
    }
    with open(os.path.join(os.path.dirname(path), "judge_model_meta.json"), "w") as handle:
        json.dump(metadata, handle)


class FutureKVJudgeTest(unittest.TestCase):
    def test_query_ring_buffer_preserves_recent_query_order(self):
        state = FutureKVLayerRuntimeState()
        expected = torch.empty(1, 0, 2)

        for values in (
            torch.arange(6, dtype=torch.float32).view(3, 1, 2),
            torch.arange(6, 12, dtype=torch.float32).view(3, 1, 2),
            torch.arange(12, 16, dtype=torch.float32).view(2, 1, 2),
        ):
            state.append_queries(values, window_size=5)
            expected = torch.cat([expected, values.transpose(0, 1)], dim=1)[:, -5:, :]
            self.assertTrue(torch.equal(state.get_queries(), expected.unsqueeze(0)))

    def test_query_ring_buffer_replaces_with_oversized_chunk(self):
        state = FutureKVLayerRuntimeState()
        values = torch.arange(24, dtype=torch.float32).view(12, 1, 2)

        state.append_queries(values[:2], window_size=5)
        state.append_queries(values[2:], window_size=5)

        self.assertTrue(
            torch.equal(state.get_queries(), values[-5:].transpose(0, 1).unsqueeze(0))
        )

    def test_qwen3vl_futurekv_compression_applies_to_current_attention(self):
        controller = object.__new__(SparseController)
        controller.is_futurekv = True
        controller.config = SimpleNamespace(
            futurekv_num_full_layers=0,
            futurekv_budget=4,
            futurekv_step_drop=2,
            futurekv_divide_length=128,
            futurekv_window_size=4,
        )
        controller.futurekv_runtime_states = {}
        controller.futurekv_judge = Mock()
        controller.futurekv_judge.enabled = True
        controller.futurekv_judge.score.side_effect = lambda _layer, key, _value, _info: (
            torch.arange(key.shape[-2], dtype=torch.float32).view(1, 1, -1)
        )

        cache_manager = Mock()
        cache_manager.get_futurekv_head_slots_and_indices.return_value = (
            torch.arange(6, dtype=torch.int32).view(1, 6),
            torch.arange(6, dtype=torch.long).view(1, 6),
        )
        cache_manager.get_futurekv_head_length.return_value = 6
        key = torch.randn(6, 1, 2)
        value = torch.randn(6, 1, 2)
        cache_manager.get_layer_kv_cache.return_value = (key, value)
        controller.cache_manager = cache_manager

        seq = Sequence([0] * 6)
        context = SimpleNamespace(is_prefill=False, seqs=[seq])
        with patch("sparsevllm.engine.sparse_controller.get_context", return_value=context):
            controller.before_layer_attention(0, torch.randn(1, 1, 2))
            cache_manager.apply_futurekv_head_keep.assert_called_once()
            controller.on_layer_attention_end(0)

        cache_manager.apply_futurekv_head_keep.assert_called_once()

    def test_attn_info_matches_hf_matmul_dtype_semantics(self):
        torch.manual_seed(7)
        query = torch.randn(1, 4, 5, 8, dtype=torch.bfloat16)
        key = torch.randn(1, 2, 9, 8, dtype=torch.bfloat16)

        actual, _ = compute_futurekv_attn_info(
            query,
            key,
            step_drop=3,
            runtime_state=FutureKVLayerRuntimeState(),
        )

        grouped_query = query.view(1, 2, 2, 5, 8)
        grouped_key = key.unsqueeze(2)
        weights = torch.matmul(grouped_query, grouped_key.transpose(3, 4)) / math.sqrt(8)
        causal = torch.full(
            (5, 5),
            torch.finfo(weights.dtype).min,
            dtype=weights.dtype,
        ).triu(1)
        mask = torch.zeros((5, 9), dtype=weights.dtype)
        mask[:, -5:] = causal
        weights = torch.softmax(
            weights + mask.view(1, 1, 1, 5, 9),
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)

        summed = weights.sum(dim=-2).transpose(-1, -2)
        maximum = weights[:, :, :, -3:, :].max(dim=-2).values.transpose(-1, -2)
        minimum = weights[:, :, :, -3:, :].min(dim=-2).values.transpose(-1, -2)

        def window_sum(window):
            return weights[:, :, :, -window:, :].sum(dim=-2).transpose(-1, -2)

        expected = torch.cat(
            [
                maximum[..., :-3, :],
                minimum[..., :-3, :],
                summed[..., :-3, :],
                window_sum(16)[..., :-3, :],
                window_sum(8)[..., :-3, :],
                window_sum(32)[..., :-3, :],
                summed[..., :-3, :],
                summed[..., :-3, :],
            ],
            dim=-1,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_load_futurekv_judge_state_from_pt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "judge_model.pt")
            save_checkpoint(path)

            by_layer = load_futurekv_judge_state(path)

        self.assertEqual(sorted(by_layer), [0, 1])
        self.assertEqual(set(by_layer[0]), {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"})

    def test_judge_bank_scores_with_local_head_slice(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "judge_model.pt")
            save_checkpoint(
                path, num_layers=1, num_kv_heads=4, total_feature_dim=32
            )
            cfg = make_futurekv_test_config(
                num_layers=1,
                hidden_size=64,
                num_attention_heads=8,
                num_key_value_heads=4,
                torch_dtype=torch.float32,
            )
            cfg.futurekv_judge_path = path

            bank = FutureKVJudgeBank(cfg, rank=1, world_size=2, device="cpu")

        self.assertTrue(bank.enabled)
        self.assertEqual(bank.local_kv_heads, 2)
        key = torch.randn(1, 2, 5, 8)
        value = torch.randn(1, 2, 5, 8)
        attn_info = torch.randn(1, 2, 5, 16)
        scores = bank.score(0, key, value, attn_info)
        self.assertEqual(tuple(scores.shape), (1, 2, 5))

    def test_compute_futurekv_attn_info_shape_and_state_gather(self):
        state = FutureKVLayerRuntimeState()
        query = torch.randn(1, 4, 3, 8)
        key = torch.randn(1, 2, 6, 8)
        attn_info, state = compute_futurekv_attn_info(
            query,
            key,
            step_drop=2,
            runtime_state=state,
        )

        self.assertEqual(tuple(attn_info.shape), (1, 2, 4, 16))
        self.assertEqual(tuple(state.attn_acc.shape), (1, 2, 6, 2))

        keep = torch.tensor([1, 3, 4, 5], dtype=torch.long)
        state.gather_keep_indices(keep)
        self.assertEqual(tuple(state.attn_acc.shape), (1, 2, 4, 2))

    def test_repeated_compression_uses_only_recent_step_queries(self):
        query = torch.randn(1, 4, 6, 8)
        key = torch.randn(1, 2, 8, 8)
        attn_acc = torch.randn(1, 2, 8, 2)
        attn_acc_decay = torch.randn(1, 2, 8, 2)
        full_state = FutureKVLayerRuntimeState(
            attn_acc=attn_acc.clone(),
            attn_acc_decay=attn_acc_decay.clone(),
        )
        suffix_state = FutureKVLayerRuntimeState(
            attn_acc=attn_acc.clone(),
            attn_acc_decay=attn_acc_decay.clone(),
        )

        full_info, full_state = compute_futurekv_attn_info(
            query,
            key,
            step_drop=2,
            runtime_state=full_state,
        )
        suffix_info, suffix_state = compute_futurekv_attn_info(
            query[:, :, -2:, :],
            key,
            step_drop=2,
            runtime_state=suffix_state,
        )

        self.assertTrue(torch.equal(full_info, suffix_info))
        self.assertTrue(torch.equal(full_state.attn_acc, suffix_state.attn_acc))
        self.assertTrue(torch.equal(full_state.attn_acc_decay, suffix_state.attn_acc_decay))

    def test_runtime_state_gathers_per_head_keep_indices(self):
        state = FutureKVLayerRuntimeState()
        state.attn_acc = torch.arange(1 * 2 * 6 * 2, dtype=torch.float32).view(1, 2, 6, 2)
        state.attn_acc_decay = state.attn_acc + 100
        keep = torch.tensor(
            [
                [0, 2, 5],
                [1, 3, 4],
            ],
            dtype=torch.long,
        )

        state.gather_keep_indices(keep)

        expected_head0 = torch.tensor([[0, 1], [4, 5], [10, 11]], dtype=torch.float32)
        expected_head1 = torch.tensor([[14, 15], [18, 19], [20, 21]], dtype=torch.float32)
        self.assertTrue(torch.equal(state.attn_acc[0, 0], expected_head0))
        self.assertTrue(torch.equal(state.attn_acc[0, 1], expected_head1))
        self.assertTrue(torch.equal(state.attn_acc_decay[0, 0], expected_head0 + 100))
        self.assertTrue(torch.equal(state.attn_acc_decay[0, 1], expected_head1 + 100))


if __name__ == "__main__":
    unittest.main()
