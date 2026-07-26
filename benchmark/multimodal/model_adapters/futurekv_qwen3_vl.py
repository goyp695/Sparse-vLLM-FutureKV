"""Qwen3-VL generation adapter for the public FutureKV evaluation path.

This module deliberately supports only two self-contained backends:

* ``hf``: the dense Hugging Face baseline;
* ``sparsevllm``: this repository's native dense/FutureKV runtime.

It must not import an external vLLM checkout or mutate ``sys.path``.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration


QWEN3VL_BACKENDS = {"hf", "sparsevllm"}


def resolve_torch_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    name = str(value).strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {value!r}") from exc


def ensure_left_padding(processor: Any) -> None:
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def batch_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def load_hf_model(
    args: Any,
    device: torch.device | str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    if str(args.method).lower() != "fullkv":
        raise ValueError("backend='hf' is the dense baseline and requires --method fullkv.")

    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    processor_path = str(args.processor_path or args.model_path)
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    ensure_left_padding(processor)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        device_map={"": str(device)} if device.type == "cuda" else None,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    return model, processor, {
        "backend": "hf",
        "model_type": "qwen3_vl",
        "method": "fullkv",
        "supports_batch_generation": True,
        "uses_sparsevllm_engine": False,
    }


class SparseQwen3VLGenerationWrapper:
    """Expose the HF ``generate`` shape over Sparse-vLLM's request API."""

    def __init__(self, engine: Any, processor: Any, config: Any, seed: int | None = None):
        self.engine = engine
        self.processor = processor
        self.seed = seed
        self.image_token_id = int(config.image_token_id)
        self.vision_start_token_id = int(config.vision_start_token_id)

    @staticmethod
    def _trim_prompt_ids(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        row_idx: int,
    ) -> list[int]:
        row = input_ids[row_idx]
        if attention_mask is not None:
            row = row[attention_mask[row_idx].to(dtype=torch.bool)]
        return [int(token_id) for token_id in row.tolist()]

    def _num_images_in_prompt(self, prompt_ids: list[int]) -> int:
        return sum(
            token_id == self.vision_start_token_id
            and prompt_ids[index + 1] == self.image_token_id
            for index, token_id in enumerate(prompt_ids[:-1])
        )

    def _build_prompts(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
    ) -> list[dict[str, Any]]:
        prompts: list[dict[str, Any]] = []
        image_cursor = 0
        patch_cursor = 0
        for row_idx in range(int(input_ids.shape[0])):
            prompt_ids = self._trim_prompt_ids(input_ids, attention_mask, row_idx)
            num_images = self._num_images_in_prompt(prompt_ids)
            multi_modal_data = None
            if num_images:
                if pixel_values is None or image_grid_thw is None:
                    raise ValueError(
                        "Qwen3-VL prompt contains image tokens but the processor "
                        "returned no image tensors."
                    )
                grid = image_grid_thw[image_cursor:image_cursor + num_images].contiguous()
                patch_count = int(grid.prod(dim=-1).sum().item())
                pixels = pixel_values[patch_cursor:patch_cursor + patch_count].contiguous()
                if len(grid) != num_images or len(pixels) != patch_count:
                    raise ValueError("Qwen3-VL image tensors do not match the prompt image tokens.")
                multi_modal_data = {
                    "pixel_values": pixels,
                    "image_grid_thw": grid,
                }
                image_cursor += num_images
                patch_cursor += patch_count
            prompts.append({
                "prompt_token_ids": prompt_ids,
                "multi_modal_data": multi_modal_data,
            })
        if image_grid_thw is not None and image_cursor != int(image_grid_thw.shape[0]):
            raise ValueError("Unused Qwen3-VL image grids remain after splitting the batch.")
        if pixel_values is not None and patch_cursor != int(pixel_values.shape[0]):
            raise ValueError("Unused Qwen3-VL image patches remain after splitting the batch.")
        return prompts

    def generate(self, **inputs: Any) -> torch.Tensor:
        from sparsevllm.sampling_params import SamplingParams

        repetition_penalty = float(inputs.get("repetition_penalty", 1.0) or 1.0)
        presence_penalty = float(inputs.get("presence_penalty", 0.0) or 0.0)
        if repetition_penalty != 1.0 or presence_penalty != 0.0:
            raise NotImplementedError(
                "Native Sparse-vLLM does not yet implement repetition/presence penalties; "
                "use their neutral values (1.0 and 0.0)."
            )
        temperature = float(inputs.get("temperature", 0.0) or 0.0)
        if temperature > 0:
            raise NotImplementedError(
                "The public Sparse-vLLM MathVision path currently requires greedy "
                "decoding (temperature=0) because per-request seeded sampling is "
                "not implemented."
            )
        input_ids = inputs["input_ids"].detach().cpu()
        if inputs.get("pixel_values_videos") is not None or inputs.get("video_grid_thw") is not None:
            raise NotImplementedError("Native Sparse-vLLM Qwen3-VL supports images, not video.")

        def cpu_tensor(name: str) -> torch.Tensor | None:
            value = inputs.get(name)
            return value.detach().cpu() if value is not None else None

        prompts = self._build_prompts(
            input_ids,
            cpu_tensor("attention_mask"),
            cpu_tensor("pixel_values"),
            cpu_tensor("image_grid_thw"),
        )
        top_k = int(inputs.get("top_k", 0) or 0)
        sampling_params = SamplingParams(
            max_tokens=int(inputs.get("max_new_tokens", 64)),
            temperature=temperature,
            top_p=float(inputs.get("top_p", 1.0) or 1.0),
            top_k=max(0, top_k),
            ignore_eos=False,
        )
        outputs = self.engine.generate(prompts, sampling_params, use_tqdm=False)
        generated = [list(item["token_ids"]) for item in outputs]
        max_gen_len = max((len(token_ids) for token_ids in generated), default=0)
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id
        generated_tensor = torch.full(
            (len(generated), max_gen_len),
            int(pad_token_id if pad_token_id is not None else 0),
            dtype=input_ids.dtype,
        )
        for row_idx, token_ids in enumerate(generated):
            if token_ids:
                generated_tensor[row_idx, :len(token_ids)] = torch.tensor(
                    token_ids, dtype=input_ids.dtype
                )
        return torch.cat([input_ids, generated_tensor], dim=1)

    def close(self) -> None:
        if hasattr(self.engine, "exit"):
            self.engine.exit()


def build_sparsevllm_engine_kwargs(args: Any) -> dict[str, Any]:
    method = str(args.method).lower()
    if method not in {"fullkv", "futurekv"}:
        raise ValueError(
            "backend='sparsevllm' supports --method fullkv or futurekv; "
            f"got {args.method!r}."
        )
    max_num_seqs = int(getattr(args, "vllm_max_num_seqs", 0) or args.batch_size)
    if max_num_seqs < int(args.batch_size):
        raise ValueError("vllm_max_num_seqs must be >= batch_size.")

    kwargs: dict[str, Any] = {
        "sparse_method": "" if method == "fullkv" else "futurekv",
        "dtype": str(args.torch_dtype),
        "max_num_seqs_in_batch": int(args.batch_size),
        "max_decoding_seqs": max_num_seqs,
        "gpu_memory_utilization": float(
            getattr(args, "vllm_gpu_memory_utilization", 0.9)
        ),
        "tensor_parallel_size": int(
            getattr(args, "vllm_tensor_parallel_size", 1) or 1
        ),
        "chunk_prefill_size": int(
            getattr(args, "sparsevllm_chunk_prefill_size", 0) or 0
        ),
    }
    max_model_len = int(getattr(args, "vllm_max_model_len", 0) or 0)
    if max_model_len > 0:
        kwargs["max_model_len"] = max_model_len
    if method == "futurekv":
        judge_path = str(getattr(args, "judge_state_path", "") or "").strip()
        if not judge_path:
            raise ValueError("--judge_state_path is required for --method futurekv.")
        if not os.path.isfile(judge_path):
            raise FileNotFoundError(f"FutureKV judge checkpoint does not exist: {judge_path}")
        kwargs.update({
            "futurekv_budget": int(args.kv_budget),
            "futurekv_window_size": int(args.window_size),
            "futurekv_step_drop": int(args.step_drop),
            "futurekv_divide_length": int(args.divide_length),
            "futurekv_num_full_layers": int(
                getattr(args, "futurekv_num_full_layers", 0) or 0
            ),
            "futurekv_judge_path": judge_path,
        })
    return kwargs


def load_sparsevllm_model(
    args: Any,
    device: torch.device | str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    del device
    from sparsevllm import LLM

    processor_path = str(args.processor_path or args.model_path)
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    ensure_left_padding(processor)
    hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    engine_kwargs = build_sparsevllm_engine_kwargs(args)
    engine = LLM(args.model_path, **engine_kwargs)
    model = SparseQwen3VLGenerationWrapper(
        engine, processor, hf_config, seed=int(args.seed)
    )
    return model, processor, {
        "backend": "sparsevllm",
        "engine": "sparsevllm.LLM",
        "model_type": "qwen3_vl",
        "method": str(args.method).lower(),
        "supports_batch_generation": True,
        "uses_sparsevllm_engine": True,
        "engine_kwargs": engine_kwargs,
    }


def load_model(
    args: Any,
    device: torch.device | str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    backend = str(getattr(args, "backend", "hf")).strip().lower()
    if backend == "hf":
        return load_hf_model(args, device=device)
    if backend == "sparsevllm":
        return load_sparsevllm_model(args, device=device)
    raise ValueError(
        f"Unsupported Qwen3-VL backend={backend!r}; expected one of "
        f"{sorted(QWEN3VL_BACKENDS)}."
    )
