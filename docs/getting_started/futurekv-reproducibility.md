# FutureKV reproducibility

Record these values for every run:

- repository commit and Python environment;
- base model and processor revisions;
- judge checkpoint pair and metadata schema;
- dataset identity, range, shard assignment, and image root;
- `budget`, `window_size`, `step_drop`, and `divide_length`;
- tensor-parallel size, batch/submission size, prefill chunk size, and memory
  utilization;
- seed and all decoding parameters.

The MathVision driver writes a sidecar `*.run_info.json` with the effective
configuration and a segment history. Results can resume by stable sample ID.
For the reproduced greedy setup, use `temperature=0`, neutral repetition and
presence penalties, and the defaults `budget=1024`, `window_size=1024`,
`step_drop=256`.

`window_size` is the query-feature ring capacity. It does not increase the
retained KV budget. This distinction is important when comparing memory use.

Before publishing results, run:

```bash
PYTHONPATH=src:futurekv_training python -m pytest -q \
  tests/test_futurekv_checkpoint_contract.py \
  tests/test_futurekv_judge.py \
  tests/test_futurekv_cache_manager.py \
  tests/test_qwen3vl_mathvision_driver.py \
  futurekv_training/tests

python scripts/validation/check_public_repo.py
```
