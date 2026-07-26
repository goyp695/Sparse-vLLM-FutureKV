import unittest

import numpy as np
import torch

from sparsevllm.engine.cache_manager.futurekv import FutureKVCacheManager
from sparsevllm.engine.cache_manager.base import SparseSelection
from sparsevllm.engine.sequence import Sequence
from unittest.mock import patch


def make_futurekv_manager(base_slots, *, num_kv_heads=2):
    manager = object.__new__(FutureKVCacheManager)
    manager.num_layers = 1
    manager.num_kv_heads = int(num_kv_heads)
    manager.futurekv_eviction_count = 0
    manager.futurekv_dropped_slots = 0
    manager.futurekv_head_states = [{}]
    manager.seq_id_to_row = [{}]
    manager.buffer_req_to_token_slots = [torch.zeros((1, 16), dtype=torch.int32)]
    manager.row_seq_lens = [np.zeros((1,), dtype=np.int32)]
    manager.free_slots_stack = [torch.zeros(32, dtype=torch.int32)]
    manager._num_free_slots = [0]
    manager.config = FakeConfig()

    slots = torch.tensor(base_slots, dtype=torch.int32)
    manager.buffer_req_to_token_slots[0][0, : slots.numel()] = slots
    manager.row_seq_lens[0][0] = int(slots.numel())
    return manager


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

    def test_per_head_keep_frees_only_slots_unused_by_all_heads(self):
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
                [0, 2, 4, 5],
                [2, 3, 4, 5],
            ],
            dtype=torch.long,
        )

        new_slots, new_indices = manager.apply_futurekv_head_keep(
            0,
            seq,
            current_slots,
            current_indices,
            keep_indices,
        )

        self.assertEqual(new_slots.tolist(), [[10, 12, 14, 15], [12, 13, 14, 15]])
        self.assertEqual(new_indices.tolist(), [[0, 2, 4, 5], [2, 3, 4, 5]])
        self.assertEqual(int(manager.row_seq_lens[0][0]), 5)
        self.assertEqual(manager.buffer_req_to_token_slots[0][0, :5].tolist(), [10, 12, 13, 14, 15])
        self.assertEqual(int(manager._num_free_slots[0]), 1)
        self.assertEqual(int(manager.free_slots_stack[0][0]), 11)
        self.assertEqual(manager.futurekv_eviction_count, 1)
        self.assertEqual(manager.futurekv_dropped_slots, 1)

        active_slots, active_indices, context_lens = manager.build_futurekv_head_read_view(0, [seq])
        slot_storage = active_slots.data_ptr()
        index_storage = active_indices.data_ptr()
        length_storage = context_lens.data_ptr()
        self.assertEqual(tuple(active_slots.shape), (1, 2, 4))
        self.assertEqual(tuple(active_indices.shape), (1, 2, 4))
        self.assertEqual(context_lens.tolist(), [4])
        self.assertEqual(active_slots[0].tolist(), [[10, 12, 14, 15], [12, 13, 14, 15]])

        active_slots, active_indices, context_lens = manager.build_futurekv_head_read_view(0, [seq])
        self.assertEqual(active_slots.data_ptr(), slot_storage)
        self.assertEqual(active_indices.data_ptr(), index_storage)
        self.assertEqual(context_lens.data_ptr(), length_storage)

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
        manager.apply_futurekv_head_keep(0, seq, current_slots, current_indices, keep_indices)

        state = manager.get_futurekv_head_state(0, seq)
        slot_storage = state.slot_buffer.data_ptr()
        index_storage = state.index_buffer.data_ptr()

        manager.buffer_req_to_token_slots[0][0, 5] = 16
        manager.row_seq_lens[0][0] = 6
        manager.sync_futurekv_head_state(0, seq, torch.tensor([99], dtype=torch.long))

        state = manager.get_futurekv_head_state(0, seq)
        self.assertIsNotNone(state)
        self.assertEqual(state.slots.tolist(), [[10, 12, 14, 15, 16], [12, 13, 14, 15, 16]])
        self.assertEqual(state.indices.tolist(), [[0, 2, 4, 5, 99], [2, 3, 4, 5, 99]])
        self.assertEqual(state.base_len, 6)
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
        manager.apply_futurekv_head_keep(0, seq, current_slots, current_indices, keep_indices)

        active_slots, active_indices, _ = manager.build_futurekv_head_read_view(0, [seq])
        slot_storage = active_slots.data_ptr()
        index_storage = active_indices.data_ptr()
        self.assertGreater(manager._futurekv_read_buffers[0][0].shape[2], active_slots.shape[2])

        manager.buffer_req_to_token_slots[0][0, 5] = 16
        manager.row_seq_lens[0][0] = 6
        manager.sync_futurekv_head_state(0, seq, torch.tensor([99], dtype=torch.long))
        active_slots, active_indices, context_lens = manager.build_futurekv_head_read_view(0, [seq])

        self.assertEqual(active_slots.data_ptr(), slot_storage)
        self.assertEqual(active_indices.data_ptr(), index_storage)
        self.assertEqual(context_lens.tolist(), [5])
        self.assertEqual(active_slots[0].tolist(), [[10, 12, 14, 15, 16], [12, 13, 14, 15, 16]])
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
            manager.apply_futurekv_head_keep(
                0,
                seq,
                current_slots,
                current_indices,
                keep_indices,
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
