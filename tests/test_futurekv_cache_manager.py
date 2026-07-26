import unittest

import numpy as np
import torch

from sparsevllm.engine.cache_manager.futurekv import FutureKVCacheManager
from sparsevllm.engine.cache_manager.base import SparseSelection
from sparsevllm.engine.sequence import Sequence
from unittest.mock import patch


def make_futurekv_manager(base_slots, *, num_kv_heads=2, head_dim=3):
    manager = object.__new__(FutureKVCacheManager)
    manager.num_layers = 1
    manager.num_kv_heads = int(num_kv_heads)
    manager.head_dim = int(head_dim)
    manager.futurekv_eviction_count = 0
    manager.futurekv_dropped_slots = 0
    manager.futurekv_head_states = [{}]
    manager.seq_id_to_row = [{}]
    manager.buffer_req_to_token_slots = [torch.zeros((1, 16), dtype=torch.int32)]
    manager.row_seq_lens = [np.zeros((1,), dtype=np.int32)]
    manager.free_slots_stack = [torch.zeros(32, dtype=torch.int32)]
    manager._num_free_slots = [0]
    manager.config = FakeConfig()
    k_cache = torch.empty((32, num_kv_heads, head_dim), dtype=torch.float32)
    v_cache = torch.empty_like(k_cache)
    for slot in range(32):
        for head in range(num_kv_heads):
            for dim in range(head_dim):
                k_cache[slot, head, dim] = slot * 100 + head * 10 + dim
                v_cache[slot, head, dim] = slot * 1000 + head * 10 + dim
    manager.kv_cache = [(k_cache, v_cache)]
    manager.get_layer_kv_cache = lambda _layer_idx: manager.kv_cache[0]

    slots = torch.tensor(base_slots, dtype=torch.int32)
    manager.buffer_req_to_token_slots[0][0, : slots.numel()] = slots
    manager.row_seq_lens[0][0] = int(slots.numel())
    return manager


def gather_head_kv(manager, head_slots):
    k_cache, v_cache = manager.get_layer_kv_cache(0)
    slots = head_slots.to(torch.long)
    heads = torch.arange(slots.shape[0], dtype=torch.long).view(-1, 1).expand_as(slots)
    return (
        k_cache[slots, heads].unsqueeze(0).contiguous(),
        v_cache[slots, heads].unsqueeze(0).contiguous(),
    )


class FakeConfig:
    snapkv_window_size = 32
    futurekv_budget = 4
    futurekv_step_drop = 2
    futurekv_divide_length = 2


class FutureKVCacheManagerTest(unittest.TestCase):
    def test_decode_compute_view_uses_per_head_slots(self):
        manager = object.__new__(FutureKVCacheManager)
        head_slots = torch.tensor([[[3, 5], [4, 5]]], dtype=torch.int32)
        context_lens = torch.tensor([2], dtype=torch.int32)
        k_cache = torch.empty((8, 2, 4))
        v_cache = torch.empty((8, 2, 4))
        manager.build_futurekv_head_read_view = lambda *args, **kwargs: (
            head_slots,
            None,
            context_lens,
        )
        manager.get_layer_kv_cache = lambda _layer_idx: (k_cache, v_cache)
        selection = SparseSelection(
            kind="full",
            req_indices=torch.tensor([0], dtype=torch.int32),
            context_lens=context_lens,
            max_context_len=2,
        )

        with patch(
            "sparsevllm.engine.cache_manager.futurekv.get_context",
            return_value=type("Context", (), {"seqs": [Sequence([1, 2])]})(),
        ):
            view = manager.build_decode_compute_view(
                0,
                torch.empty((1, 4, 4)),
                selection,
                num_heads=4,
                num_kv_heads=2,
            )

        self.assertEqual(view.backend, "futurekv_head_slots")
        self.assertIs(view.active_slots, head_slots)
        self.assertIs(view.k_cache, k_cache)
        self.assertEqual(view.context_lens.tolist(), [2])

    def test_prefill_does_not_inherit_snapkv_window_reservation(self):
        seq = Sequence(list(range(300)))
        seq.num_prefilled_tokens = 267
        manager = object.__new__(FutureKVCacheManager)
        manager.config = FakeConfig()

        self.assertEqual(manager.prefill_batched_tokens_margin(), 0)
        self.assertEqual(manager.remaining_prefill_tokens(seq), 33)

    def test_per_head_keep_repacks_disjoint_heads_to_logical_target_length(self):
        seq = Sequence([0])
        manager = make_futurekv_manager([10, 11, 12, 13, 14, 15])
        manager.seq_id_to_row[0][seq.seq_id] = 0

        current_slots = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15],
                [10, 11, 12, 13, 14, 15],
            ],
            dtype=torch.int32,
        )
        current_indices = torch.arange(6, dtype=torch.long).view(1, -1).expand(2, -1).contiguous()
        keep_indices = torch.tensor(
            [
                [0, 1, 4, 5],
                [2, 3, 4, 5],
            ],
            dtype=torch.long,
        )
        gathered_key, gathered_value = gather_head_kv(manager, current_slots)
        expanded_keep = keep_indices.view(1, 2, 4, 1).expand(
            1, 2, 4, manager.head_dim
        )
        selected_key = gathered_key.gather(2, expanded_keep)[0].transpose(0, 1).contiguous()
        selected_value = gathered_value.gather(2, expanded_keep)[0].transpose(0, 1).contiguous()

        new_slots, new_indices = manager.apply_futurekv_head_keep(
            0,
            seq,
            current_slots,
            current_indices,
            keep_indices,
            gathered_key=gathered_key,
            gathered_value=gathered_value,
        )

        self.assertEqual(new_slots.tolist(), [[10, 11, 12, 13], [10, 11, 12, 13]])
        self.assertEqual(new_indices.tolist(), [[0, 1, 4, 5], [2, 3, 4, 5]])
        self.assertEqual(int(manager.row_seq_lens[0][0]), 4)
        self.assertEqual(manager.buffer_req_to_token_slots[0][0, :4].tolist(), [10, 11, 12, 13])
        self.assertEqual(int(manager._num_free_slots[0]), 2)
        self.assertEqual(set(manager.free_slots_stack[0][:2].tolist()), {14, 15})
        self.assertEqual(manager.futurekv_eviction_count, 1)
        self.assertEqual(manager.futurekv_dropped_slots, 2)
        k_cache, v_cache = manager.get_layer_kv_cache(0)
        self.assertTrue(torch.equal(k_cache[[10, 11, 12, 13]], selected_key))
        self.assertTrue(torch.equal(v_cache[[10, 11, 12, 13]], selected_value))

        active_slots, active_indices, context_lens = manager.build_futurekv_head_read_view(0, [seq])
        slot_storage = active_slots.data_ptr()
        index_storage = active_indices.data_ptr()
        length_storage = context_lens.data_ptr()
        self.assertEqual(tuple(active_slots.shape), (1, 2, 4))
        self.assertEqual(tuple(active_indices.shape), (1, 2, 4))
        self.assertEqual(context_lens.tolist(), [4])
        self.assertEqual(active_slots[0].tolist(), [[10, 11, 12, 13], [10, 11, 12, 13]])

        active_slots, active_indices, context_lens = manager.build_futurekv_head_read_view(0, [seq])
        self.assertEqual(active_slots.data_ptr(), slot_storage)
        self.assertEqual(active_indices.data_ptr(), index_storage)
        self.assertEqual(context_lens.data_ptr(), length_storage)

    def test_repack_rejects_out_of_range_keep_before_mutating_cache(self):
        seq = Sequence([0])
        manager = make_futurekv_manager([10, 11, 12, 13, 14, 15])
        manager.seq_id_to_row[0][seq.seq_id] = 0
        current_slots = torch.tensor(
            [[10, 11, 12, 13, 14, 15], [10, 11, 12, 13, 14, 15]],
            dtype=torch.int32,
        )
        current_indices = torch.arange(6, dtype=torch.long).view(1, -1).expand(2, -1)
        keep_indices = torch.tensor([[0, 1, 4, 6], [2, 3, 4, 5]], dtype=torch.long)
        gathered_key, gathered_value = gather_head_kv(manager, current_slots)
        before_k = manager.kv_cache[0][0].clone()
        with self.assertRaisesRegex(RuntimeError, "FutureKV repack validation failed"):
            manager.apply_futurekv_head_keep(
                0,
                seq,
                current_slots,
                current_indices,
                keep_indices,
                gathered_key=gathered_key,
                gathered_value=gathered_value,
            )
        self.assertTrue(torch.equal(manager.kv_cache[0][0], before_k))
        self.assertEqual(int(manager.row_seq_lens[0][0]), 6)

    def test_sync_head_state_appends_new_decode_slot_to_each_head(self):
        seq = Sequence([0])
        manager = make_futurekv_manager([10, 11, 12, 13, 14, 15])
        manager.seq_id_to_row[0][seq.seq_id] = 0

        current_slots = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15],
                [10, 11, 12, 13, 14, 15],
            ],
            dtype=torch.int32,
        )
        current_indices = torch.arange(6, dtype=torch.long).view(1, -1).expand(2, -1).contiguous()
        keep_indices = torch.tensor([[0, 2, 4, 5], [2, 3, 4, 5]], dtype=torch.long)
        gathered_key, gathered_value = gather_head_kv(manager, current_slots)
        manager.apply_futurekv_head_keep(
            0, seq, current_slots, current_indices, keep_indices,
            gathered_key=gathered_key, gathered_value=gathered_value,
        )

        state = manager.get_futurekv_head_state(0, seq)
        slot_storage = state.slot_buffer.data_ptr()
        index_storage = state.index_buffer.data_ptr()

        manager.buffer_req_to_token_slots[0][0, 4] = 16
        manager.row_seq_lens[0][0] = 5
        manager.sync_futurekv_head_state(0, seq, torch.tensor([99], dtype=torch.long))

        state = manager.get_futurekv_head_state(0, seq)
        self.assertIsNotNone(state)
        self.assertEqual(state.slots.tolist(), [[10, 11, 12, 13, 16], [10, 11, 12, 13, 16]])
        self.assertEqual(state.indices.tolist(), [[0, 2, 4, 5, 99], [2, 3, 4, 5, 99]])
        self.assertEqual(state.base_len, 5)
        self.assertEqual(state.slot_buffer.data_ptr(), slot_storage)
        self.assertEqual(state.index_buffer.data_ptr(), index_storage)

    def test_read_view_reuses_capacity_and_copies_only_appended_tail(self):
        seq = Sequence([0])
        manager = make_futurekv_manager([10, 11, 12, 13, 14, 15])
        manager.seq_id_to_row[0][seq.seq_id] = 0
        current_slots = torch.tensor(
            [[10, 11, 12, 13, 14, 15], [10, 11, 12, 13, 14, 15]],
            dtype=torch.int32,
        )
        current_indices = torch.arange(6, dtype=torch.long).view(1, -1).expand(2, -1).contiguous()
        keep_indices = torch.tensor([[0, 2, 4, 5], [2, 3, 4, 5]], dtype=torch.long)
        gathered_key, gathered_value = gather_head_kv(manager, current_slots)
        manager.apply_futurekv_head_keep(
            0, seq, current_slots, current_indices, keep_indices,
            gathered_key=gathered_key, gathered_value=gathered_value,
        )

        active_slots, active_indices, _ = manager.build_futurekv_head_read_view(0, [seq])
        slot_storage = active_slots.data_ptr()
        index_storage = active_indices.data_ptr()
        self.assertGreater(manager._futurekv_read_buffers[0][0].shape[2], active_slots.shape[2])

        manager.buffer_req_to_token_slots[0][0, 4] = 16
        manager.row_seq_lens[0][0] = 5
        manager.sync_futurekv_head_state(0, seq, torch.tensor([99], dtype=torch.long))
        active_slots, active_indices, context_lens = manager.build_futurekv_head_read_view(0, [seq])

        self.assertEqual(active_slots.data_ptr(), slot_storage)
        self.assertEqual(active_indices.data_ptr(), index_storage)
        self.assertEqual(context_lens.tolist(), [5])
        self.assertEqual(active_slots[0].tolist(), [[10, 11, 12, 13, 16], [10, 11, 12, 13, 16]])
        self.assertEqual(active_indices[0].tolist(), [[0, 2, 4, 5, 99], [2, 3, 4, 5, 99]])

    def test_deferred_decode_state_matches_eager_state_at_compression(self):
        seq = Sequence([0])
        eager = make_futurekv_manager([10, 11, 12, 13, 14, 15])
        deferred = make_futurekv_manager([10, 11, 12, 13, 14, 15])
        for manager in (eager, deferred):
            manager.seq_id_to_row[0][seq.seq_id] = 0
            current_slots = torch.tensor(
                [[10, 11, 12, 13, 14, 15], [10, 11, 12, 13, 14, 15]],
                dtype=torch.int32,
            )
            current_indices = torch.arange(6, dtype=torch.long).view(1, -1).expand(2, -1).contiguous()
            keep_indices = torch.tensor([[0, 2, 4, 5], [2, 3, 4, 5]], dtype=torch.long)
            gathered_key, gathered_value = gather_head_kv(manager, current_slots)
            manager.apply_futurekv_head_keep(
                0,
                seq,
                current_slots,
                current_indices,
                keep_indices,
                gathered_key=gathered_key,
                gathered_value=gathered_value,
            )

        for offset, slot in enumerate((16, 17, 18), start=6):
            for manager in (eager, deferred):
                row_len = int(manager.row_seq_lens[0][0])
                manager.buffer_req_to_token_slots[0][0, row_len] = slot
                manager.row_seq_lens[0][0] = row_len + 1
            eager.sync_futurekv_head_state(0, seq, torch.tensor([offset]))
            deferred.sync_futurekv_head_state(
                0,
                seq,
                torch.tensor([offset]),
                sync_indices=False,
                sync_slots=False,
            )

            active_slots, active_indices, context_lens = deferred.build_futurekv_head_read_view(
                0,
                [seq],
                include_indices=False,
            )
            self.assertIsNone(active_indices)
            self.assertEqual(context_lens.tolist(), [4 + offset - 5])
            self.assertEqual(active_slots[0, :, -1].tolist(), [slot, slot])

        eager_slots, eager_indices = eager.get_futurekv_head_slots_and_indices(0, seq)
        deferred_slots, deferred_indices = deferred.get_futurekv_head_slots_and_indices(0, seq)
        self.assertTrue(torch.equal(deferred_slots, eager_slots))
        self.assertTrue(torch.equal(deferred_indices, eager_indices))


if __name__ == "__main__":
    unittest.main()
