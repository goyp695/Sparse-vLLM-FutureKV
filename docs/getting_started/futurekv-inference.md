# FutureKV Qwen3-VL inference

Install Sparse-vLLM from the repository root:

```bash
python -m pip install -e .
```

Run MathVision with the checked entrypoint:

```bash
scripts/futurekv/run_mathvision.sh \
  /path/to/Qwen3-VL \
  /path/to/mathvision.json \
  /path/to/images \
  /path/to/judge/judge_model.pt \
  ./outputs/mathvision_futurekv.jsonl \
  --vllm_tensor_parallel_size 1 \
  --batch_size 1
```

The shell entrypoint checks all required inputs before launching Python and
sets the reproduced defaults:

```text
kv_budget=1024
window_size=1024
step_drop=256
```

For direct Python use, choose `--backend sparsevllm --method futurekv`.
`--judge_state_path` may point at `judge_model.pt`; its sibling
`judge_model_meta.json` is mandatory. Relative dataset image paths are resolved
under `--image_root`.

The dense comparison uses `--backend hf --method fullkv`. The public adapter
does not load an external vLLM overlay or modify `PYTHONPATH`.
