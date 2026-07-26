# FutureKV judge training

The trainer is independent from the root inference package:

```bash
cd futurekv_training
python -m pip install -e '.[test]'
```

Prepare a non-empty JSON list. Each record must contain user and assistant
messages plus any image paths:

```json
[
  {
    "id": "example-1",
    "messages": [
      {"role": "user", "content": "<image> Solve the problem."},
      {"role": "assistant", "content": "Reasoning and answer"}
    ],
    "images": ["example-1.png"]
  }
]
```

Run bounded judge-only training:

```bash
futurekv-train-judge \
  --base-model /path/to/Qwen3-VL \
  --dataset /path/to/train.json \
  --image-root /path/to/images \
  --output-dir ./outputs/judge \
  --device cuda:0 \
  --max-steps 1000 \
  --budget 1024 \
  --window-size 1024 \
  --step-drop 256 \
  --divide-length 128 \
  --seed 42
```

The base Qwen3-VL model is frozen. Eager full attention provides KV and
attention traces, the trainer constructs future-attention oracle rankings,
and only the head-wise judge modules receive gradients. Outputs are written
atomically as `judge_model.pt` and `judge_model_meta.json`.

Training validates the model, dataset, image root, requested device, bounds,
and all image paths before the first optimization step. Use a model-compatible
Transformers release that returns Qwen3-VL attentions and dynamic KV cache
objects.

## Two-stage continuation

Use the stage-one checkpoint for a fresh native FutureKV generation run. Feed
that JSONL, together with the original source dataset, to
`futurekv-train-two-stage`. It filters correct boxed answers, rejects duplicate
IDs and repeated tails, writes explicit filtered/skip artifacts, validates the
stage-one checkpoint, and initializes stage two from those judge weights.

Optional Qwen3-VL LoRA SFT is exposed separately as `futurekv-train-lora`.
LoRA artifacts never replace `judge_model.pt` and are not loaded implicitly by
FutureKV inference.
