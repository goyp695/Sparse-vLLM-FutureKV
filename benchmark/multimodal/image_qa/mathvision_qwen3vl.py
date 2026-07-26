#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.multimodal.model_adapters.qwen3_vl import batch_to_device, load_model


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pluggable Qwen3-VL MathVision driver with batched decode support."
    )
    parser.add_argument("--backend", default="hf", choices=["hf", "sparsevllm"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--processor_path", default="")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument(
        "--image_root",
        default="",
        help="Base directory for relative image paths in the dataset.",
    )
    parser.add_argument("--save_path", required=True, help="Output raw jsonl path.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--shard_world_size", type=int, default=1)
    parser.add_argument(
        "--shard_assignment_path",
        default="",
        help="Optional deterministic JSON assignment used instead of modulo sharding.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--submission_batch_size",
        type=int,
        default=0,
        help="Requests submitted per engine call; zero uses batch_size. Sparse-vLLM can continuously refill up to batch_size active slots.",
    )
    parser.add_argument("--samples_per_item", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true", help="Validate dataset/task expansion without loading model.")
    parser.add_argument(
        "--system_prompt",
        default="You are a helpful assistant suitable for solving math problems.",
    )
    parser.add_argument(
        "--append_answer_instruction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the legacy step-by-step and boxed-answer instruction.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_pixels", type=int, default=1280 * 1280)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--vllm_max_model_len", type=int, default=0)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_num_seqs", type=int, default=0)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--sparsevllm_chunk_prefill_size", type=int, default=0)
    parser.add_argument("--vllm_limit_images", type=int, default=2)
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--vllm_disable_chunked_prefill", action="store_true")
    parser.add_argument("--vllm_disable_prefix_caching", action="store_true")

    parser.add_argument("--method", default="fullkv", choices=["fullkv", "futurekv"])
    parser.add_argument("--judge_state_path", default="")
    parser.add_argument("--kv_budget", type=int, default=1024)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--divide_length", type=int, default=128)
    parser.add_argument("--step_drop", type=int, default=256)
    parser.add_argument("--futurekv_num_full_layers", type=int, default=0)
    parser.add_argument("--futurekv_verbose", action="store_true")
    return parser.parse_args()


def load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"MathVision dataset must be a list, got {type(data).__name__}: {path}")
    return data


def item_id(item: dict[str, Any], idx: int) -> str:
    return str(item.get("id", idx))


def sample_uid(item: dict[str, Any], idx: int, sample_idx: int, samples_per_item: int) -> str:
    base = item_id(item, idx)
    if samples_per_item <= 1:
        return base
    return f"{base}__sample{sample_idx}"


def load_completed_ids(path: str) -> set[str]:
    completed = set()
    if not path or not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            uid = row.get("sample_uid", row.get("id"))
            if uid is not None:
                completed.add(str(uid))
    return completed


def load_image(image_path: str, max_pixels: int | None) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if max_pixels is None:
        return image
    width, height = image.size
    total_pixels = width * height
    if total_pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / total_pixels)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    try:
        resample = Image.Resampling.BICUBIC
    except AttributeError:
        resample = Image.BICUBIC
    return image.resize(new_size, resample)


def construct_messages(
    data_item: dict[str, Any],
    system_prompt: str | None,
    image_cache: dict[str, Image.Image],
    max_pixels: int | None,
    append_answer_instruction: bool = True,
    image_root: str | Path = "",
) -> list[dict[str, Any]]:
    original_content = data_item["messages"][0]["content"]
    answer_instruction = (
        "\n\nPlease reason step by step.\nPut the final answer ONLY inside \\boxed{}.\n"
        if append_answer_instruction
        else ""
    )
    user_content = []

    image_paths = list(data_item.get("images") or [])
    if image_paths:
        image_index = 0
        parts = re.split(r"(<image\d*>)", original_content)
        for part in parts:
            if re.fullmatch(r"<image\d*>", part):
                if image_index >= len(image_paths):
                    user_content.append({"type": "text", "text": part})
                    continue
                img_path = Path(image_paths[image_index])
                if not img_path.is_absolute() and image_root:
                    img_path = Path(image_root) / img_path
                img_path = str(img_path)
                image = image_cache.get(img_path)
                if image is None:
                    image = load_image(img_path, max_pixels=max_pixels)
                    image_cache[img_path] = image
                user_content.append({"type": "image", "image": image})
                image_index += 1
            elif part:
                text = part.strip()
                if text:
                    user_content.append({"type": "text", "text": text})
        if answer_instruction and user_content and user_content[-1]["type"] == "text":
            user_content[-1]["text"] += answer_instruction
        elif answer_instruction:
            user_content.append({"type": "text", "text": answer_instruction.strip()})
    else:
        text = re.sub(r"<image\d*>", "", original_content).strip()
        user_content.append({"type": "text", "text": text + answer_instruction})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": user_content})
    return messages


def select_tasks(dataset: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, int, str]]:
    dataset_size = len(dataset)
    if dataset_size == 0:
        return []
    start = max(0, min(args.start, dataset_size - 1))
    end = dataset_size - 1 if args.end < 0 else max(start, min(args.end, dataset_size - 1))
    if args.limit is not None and args.limit > 0:
        end = min(end, start + args.limit - 1)
    completed = load_completed_ids(args.save_path) if args.resume else set()
    shard_world_size = max(1, int(getattr(args, "shard_world_size", 1) or 1))
    shard_rank = int(getattr(args, "shard_rank", 0) or 0)
    if shard_rank < 0 or shard_rank >= shard_world_size:
        raise ValueError(
            "--shard_rank must be in [0, shard_world_size), got "
            f"rank={shard_rank} world_size={shard_world_size}"
        )

    assignment_path = str(getattr(args, "shard_assignment_path", "") or "")
    if assignment_path:
        with open(assignment_path, "r", encoding="utf-8") as f:
            assignment = json.load(f)
        assignment_world_size = int(assignment.get("world_size", len(assignment["assignments"])))
        assignments = assignment["assignments"]
        if assignment_world_size != shard_world_size or len(assignments) != shard_world_size:
            raise ValueError(
                "Shard assignment world size mismatch: "
                f"file={assignment_world_size}/{len(assignments)} requested={shard_world_size}"
            )
        all_indices = [int(idx) for rank_indices in assignments for idx in rank_indices]
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("Shard assignment contains duplicate dataset indices.")
        selected_indices = [
            int(idx)
            for idx in assignments[shard_rank]
            if start <= int(idx) <= end
        ]
        invalid = [idx for idx in selected_indices if idx < 0 or idx >= dataset_size]
        if invalid:
            raise ValueError(f"Shard assignment contains invalid dataset indices: {invalid[:8]}")
    else:
        selected_indices = [
            idx
            for idx in range(start, end + 1)
            if shard_world_size <= 1 or idx % shard_world_size == shard_rank
        ]

    tasks = []
    samples_per_item = max(1, int(args.samples_per_item))
    for idx in selected_indices:
        item = dataset[idx]
        for sample_idx in range(samples_per_item):
            uid = sample_uid(item, idx, sample_idx, samples_per_item)
            if uid in completed:
                continue
            tasks.append((idx, sample_idx, uid))
    return tasks


def iter_batches(items: list[Any], batch_size: int):
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _is_vllm_model(model: Any) -> bool:
    module = type(model).__module__
    return module.startswith("vllm.") or hasattr(model, "llm_engine")


def _message_images(messages: list[dict[str, Any]]) -> list[Image.Image]:
    images = []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image" and part.get("image") is not None:
                    images.append(part["image"])
    return images


def _vllm_prompt(processor: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    payload: dict[str, Any] = {"prompt": prompt}
    images = _message_images(messages)
    if images:
        payload["multi_modal_data"] = {
            "image": images[0] if len(images) == 1 else images,
        }
    return payload


def _sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        max_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        repetition_penalty=float(args.repetition_penalty),
        presence_penalty=float(args.presence_penalty),
        seed=int(args.seed),
    )


def generate_batch(
    model: Any,
    processor: Any,
    messages: list[list[dict[str, Any]]],
    args: argparse.Namespace,
    device: torch.device,
) -> list[str]:
    if _is_vllm_model(model):
        prompts = [_vllm_prompt(processor, item_messages) for item_messages in messages]
        outputs = model.generate(prompts, _sampling_params(args), use_tqdm=False)
        return [output.outputs[0].text for output in outputs]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = batch_to_device(inputs, device)
    input_len = inputs["input_ids"].shape[1]
    tokenizer = processor.tokenizer
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(float(args.temperature) > 0),
        "repetition_penalty": float(args.repetition_penalty),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    if gen_kwargs["do_sample"]:
        gen_kwargs.update(
            {
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
            }
        )
    if getattr(args, "backend", "hf") == "sparsevllm":
        gen_kwargs["presence_penalty"] = float(args.presence_penalty)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = output_ids[:, input_len:]
    return processor.batch_decode(
        new_tokens.tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _ground_truth(item: dict[str, Any]) -> Any:
    if "answer" in item:
        return item["answer"]
    messages = item.get("messages") or []
    if len(messages) > 1 and isinstance(messages[1], dict):
        return messages[1].get("content")
    return item.get("gt_answer", item.get("label", ""))


def _base_row(item: dict[str, Any], idx: int, sample_idx: int, uid: str) -> dict[str, Any]:
    return {
        "id": item_id(item, idx),
        "sample_uid": uid,
        "sample_idx": sample_idx,
        "base_id": item_id(item, idx),
        "question": item.get("messages", [{}])[0].get("content", ""),
        "gt_answer": _ground_truth(item),
    }


def _increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _run_info(args: argparse.Namespace, dataset: list[dict[str, Any]], tasks: list[tuple[int, int, str]]) -> dict[str, Any]:
    keys = (
        "backend",
        "model_path",
        "dataset_path",
        "image_root",
        "save_path",
        "batch_size",
        "submission_batch_size",
        "samples_per_item",
        "system_prompt",
        "append_answer_instruction",
        "method",
        "seed",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "presence_penalty",
        "max_pixels",
        "kv_budget",
        "step_drop",
        "window_size",
        "divide_length",
        "futurekv_num_full_layers",
        "judge_state_path",
        "vllm_max_model_len",
        "vllm_gpu_memory_utilization",
        "vllm_max_num_seqs",
        "vllm_tensor_parallel_size",
        "sparsevllm_chunk_prefill_size",
    )
    info = {
        "argv": sys.argv,
        "num_dataset_items": len(dataset),
        "num_pending_generations": len(tasks),
        "dry_run": bool(args.dry_run),
        "started_at": _now(),
    }
    info.update({key: getattr(args, key) for key in keys if hasattr(args, key)})
    return info


def _write_run_info(path: str, info: dict[str, Any]) -> None:
    with open(path + ".run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with open(path + ".run_info.segments.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    dataset = load_json(args.dataset_path)
    tasks = select_tasks(dataset, args)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        Path(args.save_path + ".run_info.segments.jsonl").unlink(missing_ok=True)

    run_info = _run_info(args, dataset, tasks)
    total_start = time.time()
    if args.dry_run:
        run_info.update(
            {
                "status": "dry_run",
                "status_counts": {},
                "finished_at": _now(),
                "total_elapsed_s": time.time() - total_start,
            }
        )
        _write_run_info(args.save_path, run_info)
        print(json.dumps(run_info, indent=2, ensure_ascii=False))
        return 0

    device = torch.device(args.device if args.backend == "hf" else "cpu")
    load_start = time.time()
    model, processor, policy = load_model(args, device=device)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    run_info["policy"] = policy
    generation_start = time.time()

    image_cache: dict[str, Image.Image] = {}
    status_counts: dict[str, int] = {}
    generated_tokens = 0
    system_prompt = str(args.system_prompt or "")
    max_pixels = int(args.max_pixels) if int(args.max_pixels) > 0 else None
    submission_batch_size = int(getattr(args, "submission_batch_size", 0) or 0)
    if submission_batch_size <= 0:
        submission_batch_size = int(args.batch_size)
    if args.backend != "sparsevllm" and submission_batch_size != int(args.batch_size):
        raise ValueError("submission_batch_size different from batch_size is currently supported only by Sparse-vLLM.")
    batches = list(iter_batches(tasks, submission_batch_size))
    mode = "a" if args.resume else "w"
    with open(save_path, mode, encoding="utf-8") as f:
        for batch in tqdm(batches, desc="mathvision", dynamic_ncols=True):
            batch_messages = []
            batch_rows = []
            for idx, sample_idx, uid in batch:
                item = dataset[idx]
                batch_messages.append(
                    construct_messages(
                        item,
                        system_prompt=system_prompt,
                        image_cache=image_cache,
                        max_pixels=max_pixels,
                        append_answer_instruction=bool(args.append_answer_instruction),
                        image_root=args.image_root,
                    )
                )
                batch_rows.append(_base_row(item, idx, sample_idx, uid))
            try:
                responses = generate_batch(model, processor, batch_messages, args, device=device)
            except Exception as exc:
                responses = ["" for _ in batch_rows]
                errors = [repr(exc) for _ in batch_rows]
                statuses = ["error" for _ in batch_rows]
            else:
                errors = [None for _ in batch_rows]
                statuses = ["success" for _ in batch_rows]

            for row, response, status, error in zip(batch_rows, responses, statuses, errors):
                response_token_count = len(
                    processor.tokenizer.encode(response, add_special_tokens=False)
                )
                row.update(
                    {
                        "response": response,
                        "generated_tokens": response_token_count,
                        "status": status,
                        "backend": args.backend,
                        "kv_method": args.method,
                    }
                )
                if error is not None:
                    row["error"] = error
                _increment(status_counts, status)
                generated_tokens += response_token_count
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    elapsed = time.time() - generation_start
    run_info.update(
        {
            "status": "completed",
            "elapsed_s": elapsed,
            "total_elapsed_s": time.time() - total_start,
            "model_load_elapsed_s": generation_start - load_start,
            "generated_tokens": generated_tokens,
            "status_counts": status_counts,
            "finished_at": _now(),
        }
    )
    _write_run_info(args.save_path, run_info)
    print(json.dumps({k: run_info[k] for k in ("status", "status_counts", "total_elapsed_s")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
