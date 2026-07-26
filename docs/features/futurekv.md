# FutureKV

FutureKV is integrated as a first-class Sparse-vLLM cache manager for
Qwen3-VL. It physically retains a different token set for every KV head. A
small learned judge scores historical tokens using key/value features and
attention statistics; recent tokens remain protected.

The reproduced defaults used by the experiments in this workspace are:

- `futurekv_budget=1024`: maximum retained KV tokens per head after eviction.
- `futurekv_step_drop=256`: number of new/recent tokens accumulated between
  eviction decisions and protected at each decision.
- `futurekv_window_size=1024`: maximum length of the separate query ring
  buffer used to calculate judge features. It is not extra KV budget.
- `futurekv_divide_length=128`: training/runtime decision segmentation value.
- `futurekv_num_full_layers=0`: apply FutureKV from the first language layer.

The query ring stores only recent queries used to estimate how historical KV
tokens are attended. When it exceeds 1024 queries, the oldest queries are
overwritten. The KV cache itself is governed independently by the 1024-token
budget.

The runtime, Qwen3-VL model, per-head prefill/decode kernels, MathVision driver,
and checkpoint validation live in the root package. Judge training lives under
`futurekv_training/` and installs independently. Neither Python package imports
the other; their only interface is:

```text
judge_model.pt
judge_model_meta.json
```

Metadata schema version 1 records model dimensions, number of layers/KV heads,
training method, dtype, FutureKV parameters, creation details, and tensor
count. Runtime loading fails before GPU model work if the pair is missing or
incompatible.

Continue with [training](../getting_started/futurekv-training.md),
[inference](../getting_started/futurekv-inference.md), and
[reproducibility](../getting_started/futurekv-reproducibility.md).
