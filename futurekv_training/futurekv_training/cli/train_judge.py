"""Train FutureKV judge heads from full-attention Qwen3-VL traces."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import random
import re
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from futurekv_training.cache import (
    build_futurekv_training_targets,
    pairwise_rank_loss,
)
from futurekv_training.checkpoint import (
    export_judge_checkpoint,
    load_judge_checkpoint,
)
from futurekv_training.data import MultimodalRecord, load_multimodal_records
from futurekv_training.modeling import JudgeCollection


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the standalone FutureKV judge on full-attention traces."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-drop", type=int, default=256)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--divide-length", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.max_steps) <= 0:
        raise ValueError("--max-steps must be positive.")
    if int(args.step_drop) <= 0:
        raise ValueError("--step-drop must be positive.")
    if int(args.max_length) <= int(args.step_drop):
        raise ValueError("--max-length must be greater than --step-drop.")
    if not Path(args.base_model).exists():
        raise FileNotFoundError(f"Base model does not exist: {args.base_model}")
    if not Path(args.dataset).is_file():
        raise FileNotFoundError(f"Dataset does not exist: {args.dataset}")
    if not Path(args.image_root).is_dir():
        raise FileNotFoundError(f"Image root does not exist: {args.image_root}")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device.index} is unavailable; count={torch.cuda.device_count()}."
            )


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _processor_messages(
    record: MultimodalRecord,
    stack: ExitStack,
) -> list[dict[str, Any]]:
    image_iter = iter(record.image_paths)
    messages = []
    for message in record.messages:
        raw_content = message["content"]
        if isinstance(raw_content, list):
            messages.append(dict(message))
            continue
        content = []
        for part in re.split(r"(<image\d*>)", str(raw_content)):
            if re.fullmatch(r"<image\d*>", part):
                try:
                    path = next(image_iter)
                except StopIteration:
                    # MathVision can repeat a numbered placeholder after the
                    # actual image. Match the inference adapter: consume each
                    # supplied image once and preserve surplus markers as text.
                    content.append({"type": "text", "text": part})
                else:
                    image = stack.enter_context(Image.open(path)).convert("RGB")
                    content.append({"type": "image", "image": image})
            elif part:
                content.append({"type": "text", "text": part})
        messages.append({"role": message.get("role", "user"), "content": content})
    try:
        extra = next(image_iter)
    except StopIteration:
        extra = None
    if extra is not None:
        raise ValueError(f"Sample {record.sample_id} has unused image path: {extra}")
    return messages


def _legacy_kv(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return tuple(zip(cache.key_cache, cache.value_cache))
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return tuple((layer.keys, layer.values) for layer in layers)
    if isinstance(cache, (tuple, list)):
        return tuple((layer[0], layer[1]) for layer in cache)
    raise RuntimeError(f"Unsupported Transformers KV cache type: {type(cache).__name__}")


def _language_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def train(args: argparse.Namespace) -> tuple[Path, Path]:
    _validate_args(args)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    records = load_multimodal_records(args.dataset, image_root=args.image_root)
    device = torch.device(args.device)
    dtype = _torch_dtype(args.dtype)
    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=bool(args.trust_remote_code)
    )
    config = AutoConfig.from_pretrained(
        args.base_model, trust_remote_code=bool(args.trust_remote_code)
    )
    language_config = _language_config(config)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=bool(args.trust_remote_code),
    ).to(device)
    base_model.eval()
    base_model.requires_grad_(False)

    judges = JudgeCollection(language_config).to(device=device, dtype=torch.float32)
    if args.init_checkpoint:
        load_judge_checkpoint(judges, args.init_checkpoint)
    optimizer = torch.optim.AdamW(judges.parameters(), lr=float(args.learning_rate))
    order = list(range(len(records)))
    random.Random(int(args.seed)).shuffle(order)

    progress = tqdm(range(int(args.max_steps)), desc="futurekv judge")
    for step in progress:
        record = records[order[step % len(order)]]
        with ExitStack() as stack:
            messages = _processor_messages(record, stack)
            inputs = processor.apply_chat_template(
                [messages],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
                max_length=int(args.max_length),
            )
            inputs = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            with torch.no_grad():
                outputs = base_model(
                    **inputs,
                    use_cache=True,
                    output_attentions=True,
                    return_dict=True,
                )
        if outputs.attentions is None or outputs.past_key_values is None:
            raise RuntimeError(
                "The base model did not return attentions/KV. Use a Transformers "
                "version with Qwen3-VL eager attention support."
            )
        layer_kv = _legacy_kv(outputs.past_key_values)
        if len(layer_kv) != len(outputs.attentions):
            raise RuntimeError("Attention and KV layer counts differ.")

        optimizer.zero_grad(set_to_none=True)
        losses = []
        for layer_idx, ((key, value), attention) in enumerate(
            zip(layer_kv, outputs.attentions)
        ):
            targets = build_futurekv_training_targets(
                key,
                value,
                attention,
                step_drop=int(args.step_drop),
            )
            predicted = judges.judge(layer_idx)(
                targets.key.float(),
                targets.value.float(),
                targets.attn_info.float(),
            ).squeeze(-1).transpose(1, 2)
            losses.append(
                pairwise_rank_loss(predicted, targets.drop_scores.float())
            )
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(judges.parameters(), float(args.gradient_clip))
        optimizer.step()
        progress.set_postfix(loss=f"{float(loss.detach()):.5f}")

    metadata = {
        "base_model": str(args.base_model),
        "hidden_size": int(language_config.hidden_size),
        "intermediate_size": int(language_config.intermediate_size),
        "num_layers": int(language_config.num_hidden_layers),
        "num_kv_heads": int(language_config.num_key_value_heads),
        "dtype": str(args.dtype),
        "training_method": "full_attention_oracle_pairwise_rank",
        "futurekv": {
            "budget": int(args.budget),
            "window_size": int(args.window_size),
            "step_drop": int(args.step_drop),
            "divide_length": int(args.divide_length),
        },
        "creation": {
            "seed": int(args.seed),
            "max_steps": int(args.max_steps),
            "dataset": str(Path(args.dataset).resolve()),
        },
    }
    return export_judge_checkpoint(judges, args.output_dir, metadata)


def main(argv: list[str] | None = None) -> int:
    paths = train(parse_args(argv))
    print(f"Saved FutureKV checkpoint: {paths[0]}")
    print(f"Saved FutureKV metadata: {paths[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
