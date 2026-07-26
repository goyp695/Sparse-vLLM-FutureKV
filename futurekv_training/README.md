# FutureKV Training

This directory is an independently installable package for training the
Qwen3-VL FutureKV judge. It does not import the inference runtime.

```bash
python -m pip install -e '.[test]'

futurekv-train-judge \
  --base-model /path/to/Qwen3-VL \
  --dataset /path/to/train.json \
  --image-root /path/to/images \
  --output-dir ./outputs/judge \
  --device cuda:0 \
  --max-steps 1000
```

The dataset is a non-empty JSON list. Each record contains a unique optional
`id`, a `messages` list with user and assistant content, and an `images` list.
Relative image paths are resolved below `--image-root`.

Training freezes the Qwen3-VL base model, obtains full-attention and KV traces,
constructs future-attention oracle rankings, and updates only one small
head-wise judge per language layer. The output is the versioned pair
`judge_model.pt` and `judge_model_meta.json`, directly consumed by native
Sparse-vLLM FutureKV inference.

## Two-stage continuation

First produce a fresh JSONL with the native inference runner and the stage-one
judge. Then filter correct, non-repeating generations and continue training:

```bash
futurekv-train-two-stage \
  --stage1-checkpoint ./outputs/stage1 \
  --raw-generations ./outputs/stage2_raw.jsonl \
  --source-dataset /path/to/train.json \
  --filtered-dataset ./outputs/stage2_filtered.json \
  --skip-report ./outputs/stage2_skips.json \
  --base-model /path/to/Qwen3-VL \
  --image-root /path/to/images \
  --output-dir ./outputs/stage2_judge \
  --device cuda:0 \
  --max-steps 500
```

The filter rejects duplicate IDs, generation errors, empty/unparseable
answers, incorrect boxed answers, and repeated tails. The stage-one checkpoint
is validated and loaded before stage-two optimization.

## Optional LoRA

LoRA is separate from judge training:

```bash
futurekv-train-lora \
  --base-model /path/to/Qwen3-VL \
  --dataset /path/to/train.json \
  --image-root /path/to/images \
  --output-dir ./outputs/lora \
  --device cuda:0 \
  --max-steps 500
```

It saves only `adapter_model.pt` and `adapter_config.json`; these are not part
of the FutureKV judge checkpoint contract.
