"""Optional Qwen3-VL LoRA supervised fine-tuning."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import random

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from futurekv_training.cli.train_judge import _processor_messages, _torch_dtype
from futurekv_training.data import load_multimodal_records
from futurekv_training.trainers.lora import inject_lora, lora_state_dict


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional standalone Qwen3-VL LoRA SFT.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    records = load_multimodal_records(args.dataset, image_root=args.image_root)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=args.trust_remote_code
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=_torch_dtype(args.dtype),
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.requires_grad_(False)
    targets = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    replaced = inject_lora(
        model,
        target_suffixes=targets,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    order = list(range(len(records)))
    random.Random(args.seed).shuffle(order)
    model.train()
    for step in tqdm(range(args.max_steps), desc="futurekv LoRA"):
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
                max_length=args.max_length,
            )
            inputs = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            labels = inputs["input_ids"].clone()
            labels[inputs.get("attention_mask", torch.ones_like(labels)) == 0] = -100
            loss = model(**inputs, labels=labels, use_cache=False).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), output / "adapter_model.pt")
    (output / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "rank": args.rank,
                "alpha": args.alpha,
                "dropout": args.dropout,
                "target_modules": list(targets),
                "replaced_modules": replaced,
                "seed": args.seed,
                "max_steps": args.max_steps,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
