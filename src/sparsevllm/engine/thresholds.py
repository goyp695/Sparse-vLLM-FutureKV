from __future__ import annotations


def sparse_base_tokens(config, *, is_prefill_budget: bool = False) -> int:
    """Return the method-specific kept-context budget used for scheduling."""

    method = str(getattr(config, "vllm_sparse_method", "") or "")
    if method == "deltakv-snapkv":
        return (
            int(config.num_sink_tokens)
            + int(config.num_recent_tokens)
            + int(config.snapkv_window_size)
        )
    if method == "deltakv-standalone":
        return int(config.num_sink_tokens) + int(config.num_recent_tokens)
    if method in ("streamingllm", "attention-sink", "attention_sink"):
        return int(config.num_sink_tokens) + int(config.num_recent_tokens)
    if method == "futurekv":
        return int(config.futurekv_budget) + int(config.futurekv_step_drop)

    num_top = (
        getattr(config, "num_top_tokens_in_prefill", None)
        if is_prefill_budget
        else getattr(config, "num_top_tokens", 0)
    )
    if num_top is None:
        num_top = getattr(config, "num_top_tokens", 0)
    return int(config.num_sink_tokens) + int(config.num_recent_tokens) + int(num_top)


def sparse_long_text_threshold(config, *, is_prefill: bool) -> int:
    base = sparse_base_tokens(config, is_prefill_budget=False)
    method = str(getattr(config, "vllm_sparse_method", "") or "")
    if is_prefill and method in ("snapkv", "pyramidkv"):
        return int(sparse_base_tokens(config, is_prefill_budget=True))
    if is_prefill:
        base += int(config.chunk_prefill_size)
    return int(base)


def sparse_decode_partitions_by_long_text(config) -> bool:
    method = str(getattr(config, "vllm_sparse_method", "") or "")
    return method != "futurekv"


def sparse_warmup_prompt_len(config) -> int:
    return (
        sparse_base_tokens(config, is_prefill_budget=True)
        + int(config.chunk_prefill_size)
        + 1024
    )
