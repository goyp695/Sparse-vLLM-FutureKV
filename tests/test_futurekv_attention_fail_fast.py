from types import SimpleNamespace
from unittest.mock import patch

import torch

from sparsevllm.engine.cache_manager.base import PrefillComputeView
from sparsevllm.layers.attention_backend import TritonAttentionBackend


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
