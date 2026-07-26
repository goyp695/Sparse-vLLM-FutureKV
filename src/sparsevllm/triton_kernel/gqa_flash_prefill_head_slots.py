import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_flash_prefill_head_slots_kernel(
    Q,
    K,
    V,
    HeadSlots,
    HeadIndices,
    O,
    q_len,
    kv_len,
    query_position_start,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_sh,
    stride_ss,
    stride_ih,
    stride_is,
    stride_om,
    stride_oh,
    stride_od,
    gqa_group_size,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    query_head = tl.program_id(1)
    kv_head = query_head // gqa_group_size

    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    query_mask = query_offsets < q_len
    query = tl.load(
        Q + query_offsets[:, None] * stride_qm + query_head * stride_qh + dim_offsets[None, :] * stride_qd,
        mask=query_mask[:, None],
        other=0.0,
    )

    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    query_positions = query_position_start + query_offsets

    for key_start in range(0, kv_len, BLOCK_N):
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        key_mask = key_offsets < kv_len
        physical_slots = tl.load(
            HeadSlots + kv_head * stride_sh + key_offsets * stride_ss,
            mask=key_mask,
            other=0,
        )
        logical_indices = tl.load(
            HeadIndices + kv_head * stride_ih + key_offsets * stride_is,
            mask=key_mask,
            other=0,
        )
        key = tl.load(
            K
            + physical_slots[None, :] * stride_ks
            + kv_head * stride_kh
            + dim_offsets[:, None] * stride_kd,
            mask=key_mask[None, :],
            other=0.0,
        )
        logits = tl.dot(query, key) * sm_scale
        causal_mask = logical_indices[None, :] <= query_positions[:, None]
        logits = tl.where(key_mask[None, :], tl.where(causal_mask, logits, -1.0e20), -float("inf"))

        block_max = tl.max(logits, axis=1)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(logits - new_max[:, None])
        value = tl.load(
            V
            + physical_slots[:, None] * stride_vs
            + kv_head * stride_vh
            + dim_offsets[None, :] * stride_vd,
            mask=key_mask[:, None],
            other=0.0,
        )
        accumulator = accumulator * old_scale[:, None] + tl.dot(probabilities.to(value.dtype), value)
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
        running_max = new_max

    output = accumulator / running_sum[:, None]
    tl.store(
        O + query_offsets[:, None] * stride_om + query_head * stride_oh + dim_offsets[None, :] * stride_od,
        output,
        mask=query_mask[:, None],
    )


@torch.no_grad()
def gqa_flash_prefill_head_slots(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    head_slots: torch.Tensor,
    head_indices: torch.Tensor,
    query_position_start: int,
    output: torch.Tensor,
):
    if query.dim() != 3:
        raise ValueError(f"query must be [tokens, heads, dim], got {tuple(query.shape)}")
    if head_slots.dim() != 2 or head_indices.shape != head_slots.shape:
        raise ValueError("head_slots and head_indices must have shape [kv_heads, tokens].")
    query_len, query_heads, head_dim = query.shape
    kv_heads, kv_len = head_slots.shape
    if query_heads % kv_heads != 0:
        raise ValueError(f"query heads must divide KV heads, got q={query_heads} kv={kv_heads}")
    if head_dim not in {16, 32, 64, 128}:
        raise ValueError(f"unsupported head_dim={head_dim}")
    if kv_len <= 0:
        raise ValueError("head-slot prefill requires at least one KV token.")

    block_m = 16
    block_n = 32
    grid = (triton.cdiv(query_len, block_m), query_heads)
    _gqa_flash_prefill_head_slots_kernel[grid](
        query,
        key_cache,
        value_cache,
        head_slots,
        head_indices,
        output,
        int(query_len),
        int(kv_len),
        int(query_position_start),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        head_slots.stride(0),
        head_slots.stride(1),
        head_indices.stride(0),
        head_indices.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        query_heads // kv_heads,
        1.0 / (head_dim ** 0.5),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=head_dim,
        num_warps=4,
        num_stages=2,
    )
