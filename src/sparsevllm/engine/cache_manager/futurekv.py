from __future__ import annotations

from dataclasses import dataclass

import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.cache_manager.base import (
    DecodeComputeView,
    PrefillComputeView,
    SparseSelection,
)
from sparsevllm.utils.context import get_context
from sparsevllm.utils.log import logger, log_level

from .snapkv import SnapKVCacheManager


@dataclass
class FutureKVHeadState:
    slot_buffer: torch.Tensor
    index_buffer: torch.Tensor
    length: int
    index_length: int
    base_len: int

    @property
    def slots(self) -> torch.Tensor:
        return self.slot_buffer[:, :self.length]

    @property
    def indices(self) -> torch.Tensor:
        return self.index_buffer[:, :self.index_length]


class FutureKVCacheManager(SnapKVCacheManager):
    """FutureKV uses SnapKV's physical slot table with a different eviction policy."""

    def __init__(self, config: Config, rank: int, world_size: int):
        self.futurekv_eviction_count = 0
        self.futurekv_dropped_slots = 0
        super().__init__(config, rank, world_size)
        self.futurekv_head_states: list[dict[int, FutureKVHeadState]] = [
            {} for _ in range(self.num_layers)
        ]
        self._futurekv_read_buffers: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = [
            None for _ in range(self.num_layers)
        ]
        self._futurekv_read_signatures: list[tuple[int, ...] | None] = [
            None for _ in range(self.num_layers)
        ]
        self._futurekv_read_lengths: list[list[int] | None] = [
            None for _ in range(self.num_layers)
        ]

    def prefill_batched_tokens_margin(self) -> int:
        return 0

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        return int(seq.num_prompt_tokens - seq.num_prefilled_tokens)

    def build_prefill_compute_view(
        self,
        layer_idx: int,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        selection: SparseSelection,
    ) -> PrefillComputeView:
        head_slots, head_indices, context_lens = self.build_futurekv_head_read_view(
            layer_idx,
            get_context().seqs,
            include_indices=True,
        )
        if head_slots is None:
            return super().build_prefill_compute_view(
                layer_idx,
                k_current,
                v_current,
                selection,
            )
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        return PrefillComputeView(
            k_cache=k_cache,
            v_cache=v_cache,
            active_slots=head_slots,
            req_indices=selection.req_indices,
            context_lens=context_lens,
            max_context_len=int(context_lens.max().item()) if context_lens.numel() else 0,
            metadata={"head_indices": head_indices},
        )

    def build_decode_compute_view(
        self,
        layer_idx: int,
        q: torch.Tensor,
        selection: SparseSelection,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> DecodeComputeView:
        head_slots, _, context_lens = self.build_futurekv_head_read_view(
            layer_idx,
            get_context().seqs,
            include_indices=False,
        )
        if head_slots is None:
            return super().build_decode_compute_view(
                layer_idx,
                q,
                selection,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        return DecodeComputeView(
            k_cache=k_cache,
            v_cache=v_cache,
            active_slots=head_slots,
            req_indices=selection.req_indices,
            context_lens=context_lens,
            max_context_len=int(context_lens.max().item()) if context_lens.numel() else 0,
            backend="futurekv_head_slots",
        )

    def free_futurekv_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
        if row_idx is None:
            raise ValueError
        before = int(self.row_seq_lens[layer_idx][row_idx])
        self.free_part_slots(layer_idx, seq, keep_indices)
        after = int(self.row_seq_lens[layer_idx][row_idx])
        dropped = max(0, before - after)
        if dropped > 0:
            self.futurekv_eviction_count += 1
            self.futurekv_dropped_slots += dropped
            if log_level == "DEBUG":
                logger.debug(
                    "[FutureKV] evicted slots: "
                    f"layer={layer_idx} seq_id={seq.seq_id} before={before} after={after} dropped={dropped}"
                )

    def free_seq(self, seq_id: int):
        seq_id = int(seq_id)
        for layer_states in getattr(self, "futurekv_head_states", []):
            layer_states.pop(seq_id, None)
        signatures = getattr(self, "_futurekv_read_signatures", None)
        lengths = getattr(self, "_futurekv_read_lengths", None)
        if signatures is not None:
            for layer_idx, signature in enumerate(signatures):
                if signature is not None and seq_id in signature:
                    signatures[layer_idx] = None
                    if lengths is not None:
                        lengths[layer_idx] = None
        return super().free_seq(seq_id)

    def _row_idx(self, layer_idx: int, seq: Sequence) -> int:
        row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
        if row_idx is None:
            raise RuntimeError(f"FutureKV row missing: layer={layer_idx} seq_id={seq.seq_id}")
        return int(row_idx)

    def _base_slots(self, layer_idx: int, row_idx: int) -> torch.Tensor:
        cur_len = int(self.row_seq_lens[layer_idx][row_idx])
        return self.buffer_req_to_token_slots[layer_idx][row_idx, :cur_len]

    def _futurekv_head_capacity(self, active_len: int) -> int:
        config = getattr(self, "config", None)
        budget = int(getattr(config, "futurekv_budget", active_len))
        step_drop = int(getattr(config, "futurekv_step_drop", 256))
        divide_length = int(getattr(config, "futurekv_divide_length", 128))
        return max(int(active_len), budget + step_drop + max(0, divide_length))

    def _make_futurekv_head_state(
        self,
        slots: torch.Tensor,
        indices: torch.Tensor,
        *,
        base_len: int,
    ) -> FutureKVHeadState:
        length = int(slots.shape[1])
        capacity = self._futurekv_head_capacity(length)
        slot_buffer = torch.empty(
            (int(slots.shape[0]), capacity),
            dtype=slots.dtype,
            device=slots.device,
        )
        index_buffer = torch.empty(
            (int(indices.shape[0]), capacity),
            dtype=indices.dtype,
            device=indices.device,
        )
        slot_buffer[:, :length].copy_(slots)
        index_buffer[:, :length].copy_(indices)
        return FutureKVHeadState(
            slot_buffer=slot_buffer,
            index_buffer=index_buffer,
            length=length,
            index_length=length,
            base_len=int(base_len),
        )

    def _ensure_futurekv_head_capacity(self, state: FutureKVHeadState, required: int):
        if required <= int(state.slot_buffer.shape[1]):
            return
        capacity = max(self._futurekv_head_capacity(required), 2 * int(state.slot_buffer.shape[1]))
        slot_buffer = torch.empty(
            (int(state.slot_buffer.shape[0]), capacity),
            dtype=state.slot_buffer.dtype,
            device=state.slot_buffer.device,
        )
        index_buffer = torch.empty(
            (int(state.index_buffer.shape[0]), capacity),
            dtype=state.index_buffer.dtype,
            device=state.index_buffer.device,
        )
        slot_buffer[:, :state.length].copy_(state.slots)
        index_buffer[:, :state.index_length].copy_(state.indices)
        state.slot_buffer = slot_buffer
        state.index_buffer = index_buffer

    def _materialize_pending_indices(self, state: FutureKVHeadState):
        pending = int(state.length) - int(state.index_length)
        if pending <= 0:
            return
        if state.index_length > 0:
            starts = state.index_buffer[:, state.index_length - 1:state.index_length] + 1
        else:
            starts = torch.zeros(
                (int(state.index_buffer.shape[0]), 1),
                dtype=state.index_buffer.dtype,
                device=state.index_buffer.device,
            )
        offsets = torch.arange(
            pending,
            dtype=state.index_buffer.dtype,
            device=state.index_buffer.device,
        ).view(1, -1)
        state.index_buffer[:, state.index_length:state.length].copy_(starts + offsets)
        state.index_length = int(state.length)

    def _materialize_pending_slots(
        self,
        layer_idx: int,
        seq: Sequence,
        state: FutureKVHeadState,
    ):
        row_idx = self._row_idx(layer_idx, seq)
        base_slots = self._base_slots(layer_idx, row_idx)
        row_len = int(base_slots.numel())
        pending = row_len - int(state.base_len)
        if pending <= 0:
            state.base_len = row_len
            return
        old_len = int(state.length)
        new_len = old_len + pending
        self._ensure_futurekv_head_capacity(state, new_len)
        new_slots = base_slots[state.base_len:row_len].to(
            device=state.slot_buffer.device,
            dtype=state.slot_buffer.dtype,
        )
        state.slot_buffer[:, old_len:new_len].copy_(
            new_slots.view(1, -1).expand(int(state.slot_buffer.shape[0]), -1)
        )
        state.length = new_len
        state.base_len = row_len

    @torch.no_grad()
    def sync_futurekv_head_state(
        self,
        layer_idx: int,
        seq: Sequence,
        append_positions: torch.Tensor | None,
        *,
        sync_indices: bool = True,
        sync_slots: bool = True,
    ):
        state = self.futurekv_head_states[layer_idx].get(int(seq.seq_id))
        if state is None:
            return
        row_idx = self._row_idx(layer_idx, seq)
        base_slots = self._base_slots(layer_idx, row_idx)
        row_len = int(base_slots.numel())
        base_len = int(state.base_len)
        if row_len <= base_len:
            state.base_len = row_len
            return
        if not sync_slots:
            return

        new_slots = base_slots[base_len:row_len].to(device=state.slots.device, dtype=state.slots.dtype)
        add_len = int(new_slots.numel())
        h = int(state.slot_buffer.shape[0])
        old_len = int(state.length)
        new_len = old_len + add_len
        self._ensure_futurekv_head_capacity(state, new_len)
        state.slot_buffer[:, old_len:new_len].copy_(new_slots.view(1, -1).expand(h, -1))
        state.length = new_len
        state.base_len = row_len
        if sync_indices:
            self._materialize_pending_indices(state)
            if append_positions is not None and int(append_positions.numel()) >= add_len:
                new_indices_1d = append_positions.reshape(-1)[-add_len:].to(
                    device=state.index_buffer.device,
                    dtype=state.index_buffer.dtype,
                )
                state.index_buffer[:, old_len:new_len].copy_(
                    new_indices_1d.view(1, -1).expand(h, -1)
                )

    def get_futurekv_head_length(self, layer_idx: int, seq: Sequence) -> int:
        state = self.get_futurekv_head_state(layer_idx, seq)
        if state is not None:
            row_len = int(self._base_slots(layer_idx, self._row_idx(layer_idx, seq)).numel())
            return int(state.length) + max(0, row_len - int(state.base_len))
        return int(self._base_slots(layer_idx, self._row_idx(layer_idx, seq)).numel())

    def get_futurekv_head_state(
        self,
        layer_idx: int,
        seq: Sequence,
    ) -> FutureKVHeadState | None:
        return self.futurekv_head_states[layer_idx].get(int(seq.seq_id))

    def get_futurekv_head_slots_and_indices(
        self,
        layer_idx: int,
        seq: Sequence,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.get_futurekv_head_state(layer_idx, seq)
        if state is not None:
            self._materialize_pending_slots(layer_idx, seq, state)
            self._materialize_pending_indices(state)
            return state.slots, state.indices

        row_idx = self._row_idx(layer_idx, seq)
        base_slots = self._base_slots(layer_idx, row_idx).to(torch.int32)
        h = int(self.num_kv_heads)
        slots = base_slots.view(1, -1).expand(h, -1).contiguous()
        indices = torch.arange(
            int(base_slots.numel()),
            device=base_slots.device,
            dtype=torch.long,
        ).view(1, -1).expand(h, -1).contiguous()
        return slots, indices

    @torch.no_grad()
    def apply_futurekv_head_keep(
        self,
        layer_idx: int,
        seq: Sequence,
        current_head_slots: torch.Tensor,
        current_head_indices: torch.Tensor,
        keep_indices: torch.Tensor,
        *,
        gathered_key: torch.Tensor,
        gathered_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_idx = self._row_idx(layer_idx, seq)
        before = int(self.row_seq_lens[layer_idx][row_idx])
        old_base_slots = self.buffer_req_to_token_slots[layer_idx][row_idx, :before].clone()

        def validation_failed(reason: str):
            raise RuntimeError(
                "FutureKV repack validation failed: "
                f"layer={layer_idx} seq_id={seq.seq_id} {reason}"
            )

        if current_head_slots.ndim != 2:
            validation_failed("current_head_slots must be shaped [heads, length].")
        if current_head_indices.shape != current_head_slots.shape:
            validation_failed("current_head_indices must match current_head_slots.")
        if keep_indices.ndim != 2 or int(keep_indices.shape[0]) != int(current_head_slots.shape[0]):
            validation_failed("keep_indices must be shaped [heads, target_length].")
        if int(current_head_slots.shape[0]) != int(self.num_kv_heads):
            validation_failed("current head count does not match the cache manager.")
        expected_gather_shape = (
            1,
            int(current_head_slots.shape[0]),
            int(current_head_slots.shape[1]),
        )
        if (
            gathered_key.ndim != 4
            or tuple(gathered_key.shape[:3]) != expected_gather_shape
            or gathered_value.shape != gathered_key.shape
        ):
            validation_failed(
                "gathered_key/value must be shaped [1, heads, current_length, head_dim]."
            )
        if gathered_key.dtype != gathered_value.dtype:
            validation_failed("gathered_key/value must use the same dtype.")
        devices = {
            current_head_slots.device,
            current_head_indices.device,
            keep_indices.device,
            gathered_key.device,
            gathered_value.device,
            old_base_slots.device,
        }
        if len(devices) != 1:
            validation_failed("all inputs must be on the same device.")

        target_len = int(keep_indices.shape[1])
        logical_len = int(current_head_slots.shape[1])
        if target_len <= 0:
            validation_failed("target length must be positive.")
        if target_len > logical_len or target_len > before:
            validation_failed("target length exceeds logical or physical cache length.")
        if int(torch.unique(old_base_slots).numel()) != before:
            validation_failed("cache-manager row must contain unique physical slots.")

        keep_indices = keep_indices.to(dtype=torch.long).contiguous()
        if int(keep_indices.min().item()) < 0 or int(keep_indices.max().item()) >= logical_len:
            validation_failed("keep_indices must be within the current logical cache length.")

        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        if (
            k_cache.ndim != 3
            or v_cache.shape != k_cache.shape
            or tuple(k_cache.shape[1:]) != (
                int(current_head_slots.shape[0]),
                int(gathered_key.shape[-1]),
            )
            or k_cache.device != gathered_key.device
            or v_cache.device != gathered_value.device
            or k_cache.dtype != gathered_key.dtype
            or v_cache.dtype != gathered_value.dtype
        ):
            validation_failed("cache shape/device/dtype does not match gathered K/V.")
        if (
            int(old_base_slots.min().item()) < 0
            or int(old_base_slots.max().item()) >= int(k_cache.shape[0])
        ):
            validation_failed("cache-manager row contains an invalid physical slot.")

        destination_slots = old_base_slots[:target_len].clone()
        expanded_keep = keep_indices.view(
            1,
            int(keep_indices.shape[0]),
            target_len,
            1,
        ).expand(
            int(gathered_key.shape[0]),
            int(keep_indices.shape[0]),
            target_len,
            int(gathered_key.shape[-1]),
        )
        selected_key = gathered_key.gather(2, expanded_keep)[0].transpose(0, 1).contiguous()
        selected_value = gathered_value.gather(2, expanded_keep)[0].transpose(0, 1).contiguous()
        destination_indices = destination_slots.to(dtype=torch.long)
        k_cache.index_copy_(0, destination_indices, selected_key)
        v_cache.index_copy_(0, destination_indices, selected_value)

        new_head_slots = destination_slots.view(1, -1).expand(
            int(current_head_slots.shape[0]), -1
        ).to(torch.int32).contiguous()
        new_head_indices = current_head_indices.gather(1, keep_indices).to(torch.long).contiguous()
        dropped_slots = old_base_slots[target_len:]

        if dropped_slots.numel() > 0:
            count = int(dropped_slots.numel())
            ptr = self._num_free_slots[layer_idx]
            self.free_slots_stack[layer_idx][ptr: ptr + count] = dropped_slots
            self._num_free_slots[layer_idx] += count
            self.futurekv_eviction_count += 1
            self.futurekv_dropped_slots += count

        self.buffer_req_to_token_slots[layer_idx][row_idx, :] = 0
        self.buffer_req_to_token_slots[layer_idx][row_idx, :target_len] = destination_slots
        self.row_seq_lens[layer_idx][row_idx] = target_len
        self.futurekv_head_states[layer_idx][int(seq.seq_id)] = self._make_futurekv_head_state(
            new_head_slots,
            new_head_indices,
            base_len=target_len,
        )

        if log_level == "DEBUG":
            logger.debug(
                "[FutureKV] per-head eviction: "
                f"layer={layer_idx} seq_id={seq.seq_id} base_before={before} "
                f"base_after={target_len} head_len={int(new_head_slots.shape[1])} "
                f"dropped={int(dropped_slots.numel())}"
            )
        return new_head_slots, new_head_indices

    def _get_futurekv_read_buffers(
        self,
        layer_idx: int,
        *,
        batch: int,
        heads: int,
        max_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        buffers_by_layer = getattr(self, "_futurekv_read_buffers", None)
        if buffers_by_layer is None:
            buffers_by_layer = [None for _ in range(self.num_layers)]
            self._futurekv_read_buffers = buffers_by_layer
        buffers = buffers_by_layer[layer_idx]
        needs_allocation = buffers is None
        if buffers is not None:
            slot_buffer, _, length_buffer = buffers
            needs_allocation = (
                int(slot_buffer.shape[0]) < batch
                or int(slot_buffer.shape[1]) < heads
                or int(slot_buffer.shape[2]) < max_len
                or int(length_buffer.shape[0]) < batch
            )
        if needs_allocation:
            old_batch = int(buffers[0].shape[0]) if buffers is not None else 0
            old_len = int(buffers[0].shape[2]) if buffers is not None else 0
            capacity_batch = max(batch, old_batch)
            capacity_len = max(
                self._futurekv_head_capacity(max_len),
                max_len,
                2 * old_len,
            )
            buffers = (
                torch.empty(
                    (capacity_batch, heads, capacity_len),
                    dtype=torch.int32,
                    device=device,
                ),
                torch.empty(
                    (capacity_batch, heads, capacity_len),
                    dtype=torch.long,
                    device=device,
                ),
                torch.empty((capacity_batch,), dtype=torch.int32, device=device),
            )
            buffers_by_layer[layer_idx] = buffers
        return buffers

    def build_futurekv_head_read_view(
        self,
        layer_idx: int,
        seqs: list[Sequence],
        *,
        include_indices: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        states = [self.get_futurekv_head_state(layer_idx, seq) for seq in seqs]
        if not any(state is not None for state in states):
            signatures = getattr(self, "_futurekv_read_signatures", None)
            lengths_by_layer = getattr(self, "_futurekv_read_lengths", None)
            if signatures is not None:
                signatures[layer_idx] = None
            if lengths_by_layer is not None:
                lengths_by_layer[layer_idx] = None
            return None, None, None

        lengths = []
        fallback = []
        for seq, state in zip(seqs, states):
            if state is None:
                slots, indices = self.get_futurekv_head_slots_and_indices(layer_idx, seq)
                fallback.append((slots, indices if include_indices else None, None))
                lengths.append(int(slots.shape[1]))
            else:
                if include_indices:
                    self._materialize_pending_slots(layer_idx, seq, state)
                    self._materialize_pending_indices(state)
                    pending_slots = None
                else:
                    row_idx = self._row_idx(layer_idx, seq)
                    base_slots = self._base_slots(layer_idx, row_idx)
                    pending_slots = base_slots[state.base_len:].to(
                        device=state.slot_buffer.device,
                        dtype=state.slot_buffer.dtype,
                    )
                fallback.append(
                    (state.slots, state.indices if include_indices else None, pending_slots)
                )
                lengths.append(
                    int(state.slots.shape[1])
                    + (int(pending_slots.numel()) if pending_slots is not None else 0)
                )

        batch = len(seqs)
        h = int(self.num_kv_heads)
        max_len = max(lengths) if lengths else 0
        device = self.buffer_req_to_token_slots[layer_idx].device
        buffers_by_layer = getattr(self, "_futurekv_read_buffers", None)
        old_buffers = buffers_by_layer[layer_idx] if buffers_by_layer is not None else None
        slot_buffer, index_buffer, length_buffer = self._get_futurekv_read_buffers(
            layer_idx,
            batch=batch,
            heads=h,
            max_len=max_len,
            device=device,
        )
        signatures = getattr(self, "_futurekv_read_signatures", None)
        if signatures is None:
            signatures = [None for _ in range(self.num_layers)]
            self._futurekv_read_signatures = signatures
        lengths_by_layer = getattr(self, "_futurekv_read_lengths", None)
        if lengths_by_layer is None:
            lengths_by_layer = [None for _ in range(self.num_layers)]
            self._futurekv_read_lengths = lengths_by_layer

        signature = tuple(int(seq.seq_id) for seq in seqs)
        previous_lengths = lengths_by_layer[layer_idx]
        full_refresh = (
            include_indices
            or old_buffers is not self._futurekv_read_buffers[layer_idx]
            or signatures[layer_idx] != signature
            or previous_lengths is None
            or len(previous_lengths) != batch
            or any(cur_len < old_len for cur_len, old_len in zip(lengths, previous_lengths))
        )
        active_slots = slot_buffer[:batch, :, :max_len]
        active_indices = index_buffer[:batch, :, :max_len] if include_indices else None
        context_lens = length_buffer[:batch]
        context_lens.copy_(torch.as_tensor(lengths, dtype=torch.int32, device=device))

        if full_refresh:
            active_slots.fill_(-1)
            if active_indices is not None:
                active_indices.fill_(-1)
            for b_idx, (slots, indices, pending_slots) in enumerate(fallback):
                stored_len = int(slots.shape[1])
                active_slots[b_idx, :, :stored_len].copy_(slots)
                if pending_slots is not None and pending_slots.numel() > 0:
                    pending_len = int(pending_slots.numel())
                    active_slots[b_idx, :, stored_len:stored_len + pending_len].copy_(
                        pending_slots.view(1, -1).expand(h, -1)
                    )
                if active_indices is not None:
                    active_indices[b_idx, :, :stored_len].copy_(indices)
        else:
            previous_max_len = max(previous_lengths, default=0)
            if max_len > previous_max_len:
                active_slots[:, :, previous_max_len:max_len].fill_(-1)
                if active_indices is not None:
                    active_indices[:, :, previous_max_len:max_len].fill_(-1)
            for b_idx, (slots, indices, pending_slots) in enumerate(fallback):
                old_len = int(previous_lengths[b_idx])
                cur_len = int(slots.shape[1])
                if old_len < cur_len:
                    active_slots[b_idx, :, old_len:cur_len].copy_(slots[:, old_len:cur_len])
                if pending_slots is not None:
                    pending_offset = max(0, old_len - cur_len)
                    if pending_offset < int(pending_slots.numel()):
                        pending_tail = pending_slots[pending_offset:]
                        active_slots[
                            b_idx,
                            :,
                            cur_len + pending_offset:cur_len + int(pending_slots.numel()),
                        ].copy_(pending_tail.view(1, -1).expand(h, -1))
                if active_indices is not None and cur_len > old_len:
                    active_indices[b_idx, :, old_len:cur_len].copy_(indices[:, old_len:cur_len])

        signatures[layer_idx] = signature
        lengths_by_layer[layer_idx] = list(lengths)
        return active_slots, active_indices, context_lens

    def free_slot_stats(self) -> dict[str, int]:
        stats = super().free_slot_stats()
        stats.update(
            {
                "futurekv_evictions": int(self.futurekv_eviction_count),
                "futurekv_dropped_slots": int(self.futurekv_dropped_slots),
            }
        )
        return stats
