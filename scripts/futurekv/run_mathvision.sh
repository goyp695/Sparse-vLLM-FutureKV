#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 MODEL_PATH DATASET_PATH IMAGE_ROOT JUDGE_MODEL OUTPUT_JSONL [extra args...]" >&2
}

if [[ $# -lt 5 ]]; then
  usage
  exit 2
fi

model_path=$1
dataset_path=$2
image_root=$3
judge_model=$4
output_jsonl=$5
shift 5

for required_dir in "$model_path" "$image_root"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "Required directory does not exist: $required_dir" >&2
    exit 2
  fi
done
for required_file in "$dataset_path" "$judge_model"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required file does not exist: $required_file" >&2
    exit 2
  fi
done
if [[ -z "$output_jsonl" ]]; then
  echo "OUTPUT_JSONL must not be empty." >&2
  exit 2
fi

python -m benchmark.multimodal.image_qa.mathvision_qwen3vl \
  --backend sparsevllm \
  --method futurekv \
  --model_path "$model_path" \
  --dataset_path "$dataset_path" \
  --image_root "$image_root" \
  --judge_state_path "$judge_model" \
  --save_path "$output_jsonl" \
  --kv_budget 1024 \
  --window_size 1024 \
  --step_drop 256 \
  "$@"
