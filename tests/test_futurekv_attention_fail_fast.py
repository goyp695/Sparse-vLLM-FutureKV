from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytest

from sparsevllm.engine.cache_manager.base import DecodeComputeView, PrefillComputeView
from sparsevllm.layers.attention_backend import TritonAttentionBackend
from sparsevllm.triton_kernel.gqa_flash_prefill_head_slots import (
    gqa_flash_prefill_head_slots,
)


def test_futurekv_per_head_prefill_uses_triton_kernel():
    view = PrefillComputeView(
        k_cache=torch.empty((1, 1, 16)),
        v_cache=torch.empty((1, 1, 16)),
        active_slots=torch.tensor([[[0]]], dtype=torch.int32),
        req_indices=torch.tensor([0], dtype=torch.int32),
        context_lens=torch.tensor([1], dtype=torch.int32),
        max_context_len=1,
        metadata={"head_indices": torch.tensor([[[0]]], dtype=torch.long)},
    )
    context = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 1]),
        seqs=[SimpleNamespace(num_prefilled_tokens=0)],
    )
    backend = TritonAttentionBackend()

    with (
        patch("sparsevllm.layers.attention_backend.get_context", return_value=context),
        patch(
            "sparsevllm.layers.attention_backend.gqa_flash_prefill_head_slots"
        ) as prefill_kernel,
    ):
        output = backend.run_prefill(
            torch.zeros((1, 1, 16)),
            view,
            b_start_loc=torch.tensor([0], dtype=torch.int32),
            chunk_lens=torch.tensor([1], dtype=torch.int32),
            max_input_len=1,
        )

    prefill_kernel.assert_called_once()
    assert output.shape == (1, 1, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_futurekv_per_head_prefill_kernel_matches_torch():
    torch.manual_seed(11)
    device = torch.device("cuda")
    query_len, kv_len, query_heads, kv_heads, head_dim = 5, 97, 4, 2, 128
    query_start = 4
    query = torch.randn(
        query_len, query_heads, head_dim, device=device, dtype=torch.float16
    )
    key = torch.randn(kv_len, kv_heads, head_dim, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    head_slots = torch.stack([
        torch.arange(kv_len, device=device, dtype=torch.int32),
        torch.arange(kv_len - 1, -1, -1, device=device, dtype=torch.int32),
    ])
    head_indices = torch.stack([
        torch.arange(kv_len, device=device),
        torch.arange(kv_len - 1, -1, -1, device=device),
    ])
    output = torch.empty_like(query)

    gqa_flash_prefill_head_slots(
        query,
        key,
        value,
        head_slots,
        head_indices,
        query_start,
        output,
    )

    expected = torch.empty_like(output)
    group_size = query_heads // kv_heads
    for query_head in range(query_heads):
        kv_head = query_head // group_size
        slots = head_slots[kv_head].long()
        gathered_key = key[slots, kv_head].float()
        gathered_value = value[slots, kv_head].float()
        logits = query[:, query_head].float() @ gathered_key.T / head_dim**0.5
        mask = head_indices[kv_head].view(1, -1) <= (
            query_start + torch.arange(query_len, device=device)
        ).view(-1, 1)
        probabilities = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
        expected[:, query_head] = (probabilities @ gathered_value).to(expected.dtype)

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_futurekv_per_head_decode_kernel_matches_torch():
    torch.manual_seed(13)
    device = torch.device("cuda")
    batch, kv_len, query_heads, kv_heads, head_dim = 1, 97, 4, 2, 128
    query = torch.randn(
        batch, query_heads, head_dim, device=device, dtype=torch.float16
    )
    key = torch.randn(kv_len, kv_heads, head_dim, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    head_slots = torch.stack([
        torch.arange(kv_len, device=device, dtype=torch.int32),
        torch.arange(kv_len - 1, -1, -1, device=device, dtype=torch.int32),
    ]).unsqueeze(0)
    context_lens = torch.tensor([kv_len], device=device, dtype=torch.int32)
    view = DecodeComputeView(
        k_cache=key,
        v_cache=value,
        active_slots=head_slots,
        req_indices=torch.tensor([0], device=device, dtype=torch.int32),
        context_lens=context_lens,
        max_context_len=kv_len,
        backend="futurekv_head_slots",
    )
    block_seq = 256
    mid_output = torch.empty(
        batch, query_heads, 1, head_dim, device=device, dtype=query.dtype
    )
    mid_logsumexp = torch.empty(
        batch, query_heads, 1, device=device, dtype=torch.float32
    )

    output = TritonAttentionBackend().run_decode(
        query,
        view,
        mid_o=mid_output,
        mid_o_logexpsum=mid_logsumexp,
        max_len_in_batch=kv_len,
        block_seq=block_seq,
        num_heads=query_heads,
        num_kv_heads=kv_heads,
    )

    expected = torch.empty_like(output)
    group_size = query_heads // kv_heads
    for query_head in range(query_heads):
        kv_head = query_head // group_size
        slots = head_slots[0, kv_head].long()
        logits = (
            query[0, query_head].float()
            @ key[slots, kv_head].float().T
            / head_dim**0.5
        )
        probabilities = torch.softmax(logits, dim=-1)
        expected[0, query_head] = (
            probabilities @ value[slots, kv_head].float()
        ).to(expected.dtype)
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
