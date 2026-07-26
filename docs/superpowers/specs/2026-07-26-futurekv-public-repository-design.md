# FutureKV Public Repository Design

## Goal

Create a clean, independently publishable repository that preserves the
complete Sparse-vLLM engine and its upstream history, adds the current native
Qwen3-VL FutureKV implementation, and includes a separately installable
FutureKV training subsystem.

The result must support the full path from FutureKV judge training to
Sparse-vLLM inference and MathVision evaluation without depending on the
original external training, evaluation, or active development checkouts.

## Repository Source and Isolation

- Use a clean Sparse-vLLM checkout as the repository base so upstream history,
  attribution, and Apache-2.0 licensing remain intact.
- Treat the active Sparse-vLLM development checkout as a read-only source for
  the current FutureKV runtime changes.
- Treat the external training and reference-evaluation trees as read-only
  sources for the current training implementation.
- Make every edit, cleanup, bug fix, and verification change only in the new
  repository.
- Do not copy logs, generated results, PID files, model weights, datasets,
  Python caches, local environments, or machine-specific temporary files.

## Repository Architecture

The repository remains a complete Sparse-vLLM distribution. FutureKV inference
is integrated as a first-class sparse method in the existing runtime, while
training is kept in a standalone top-level Python project.

```text
Sparse-vLLM-FutureKV/
├── src/
│   ├── sparsevllm/                 # Sparse-vLLM runtime and FutureKV inference
│   └── deltakv/                    # Existing DeltaKV subsystem
├── futurekv_training/
│   ├── futurekv_training/          # Standalone FutureKV training package
│   │   ├── modeling.py
│   │   ├── cache.py
│   │   ├── data.py
│   │   ├── checkpoint.py
│   │   └── trainers/
│   ├── configs/
│   ├── scripts/
│   ├── tests/
│   ├── pyproject.toml
│   └── README.md
├── benchmark/                      # Existing benchmarks plus Qwen3-VL MathVision
├── configs/futurekv/               # Reproducible FutureKV inference configs
├── scripts/futurekv/               # Public inference/evaluation entrypoints
├── tests/                          # Runtime and training/runtime contract tests
├── README.md
├── LICENSE
└── pyproject.toml
```

## Inference Subsystem

FutureKV is registered like the other Sparse-vLLM sparse methods.

- Method configuration belongs in `src/sparsevllm/config.py`.
- Method registration and prefill policy belong in
  `src/sparsevllm/method_registry.py`.
- Persistent per-request and per-head cache state belongs in
  `src/sparsevllm/engine/cache_manager/futurekv.py`.
- Judge checkpoint loading and causal attention-feature construction belong in
  `src/sparsevllm/engine/futurekv_judge.py`.
- Cross-layer selection coordination belongs in
  `src/sparsevllm/engine/sparse_controller.py`.
- Generic attention code only invokes shared cache-manager hooks.
- Qwen3-VL model support lives under `src/sparsevllm/models/`.
- The MathVision driver lives under `benchmark/multimodal/image_qa/`.

The public command must accept explicit model, dataset, output, checkpoint,
seed, KV budget, drop step, window size, tensor parallel size, and decoding
parameters. Repository scripts may provide examples but must not contain
server-specific absolute paths.

## Independent Training Subsystem

`futurekv_training` is independently installable:

```bash
pip install -e ./futurekv_training
```

The root package remains independently installable:

```bash
pip install -e .
```

Neither installation implicitly installs the other. The runtime must not import
the training package, and the training package must not import Sparse-vLLM
runtime internals.

The training subsystem includes:

- FutureKV judge architecture and oracle-label construction.
- Dataset validation and multimodal sample loading.
- Judge-only training.
- Two-stage training with fresh compressed-trajectory generation.
- The existing optional LoRA adaptation workflow when it is required to
  reproduce the current experiments.
- Checkpoint export with explicit metadata.
- Small smoke configurations that do not require publishing private datasets
  or model weights.

Training scripts take paths and hyperparameters through CLI options or checked
configuration files. Generated datasets, checkpoints, and logs go to
user-selected output directories and are ignored by Git by default.

## Checkpoint Contract

Training and inference communicate only through two versioned artifacts:

- `judge_model.pt`: CPU-loadable tensor state for each layer's
  `judge_model.fc1` and `judge_model.fc2` parameters.
- `judge_model_meta.json`: schema version, base-model identifier, architecture
  dimensions, layer count, KV-head count, dtype, training method, relevant
  FutureKV settings, and creation command/config.

The inference loader must:

- Load a single file or a documented checkpoint directory.
- Validate all required tensors, layers, head counts, and shapes.
- Support tensor-parallel head slicing.
- Fail with a clear error on incompatible or incomplete checkpoints.
- Never silently fall back to an untrained or random judge.

The training exporter must write atomically and must not claim success unless
both state and metadata validate.

## Data and Execution Flow

1. The user installs the standalone training project.
2. A validated dataset and base Qwen3-VL model are passed to the training CLI.
3. Judge-only or two-stage training produces a versioned checkpoint pair.
4. A contract validator verifies checkpoint structure without loading the full
   language model.
5. The user installs the root Sparse-vLLM package.
6. The FutureKV inference command receives the base model and judge checkpoint.
7. Sparse-vLLM loads the judge per tensor-parallel rank, performs dynamic
   per-head KV selection, and writes raw generation results.
8. MathVision evaluation parses raw output separately and writes per-sample and
   aggregate results.

## Failure Handling and Reproducibility

- Missing datasets, model paths, checkpoints, dependencies, or evaluation
  resources fail before a GPU run starts.
- Unsupported FutureKV configuration combinations fail during configuration
  validation.
- Every evaluated sample records an explicit status.
- Raw generations, parsed answers, per-sample scores, aggregate metrics, and
  run metadata remain separate artifacts.
- Retries and generation loops are bounded.
- Run metadata records the model, dataset split, prompt settings, decoding
  parameters, seed, sample range, tensor parallel size, FutureKV settings, and
  checkpoint metadata.
- Examples use placeholders or relative paths rather than local machine paths.

## Public Repository Hygiene

- Extend `.gitignore` for checkpoints, adapters, datasets, generated results,
  logs, PID files, caches, local environments, and profiler artifacts.
- Scan tracked files for secrets, usernames, `/home/`, `/media/`, internal IPs,
  API keys, and stale absolute paths.
- Keep the Apache-2.0 license and clearly credit Sparse-vLLM and other inherited
  components.
- Document which code is inherited and which functionality adds FutureKV.
- Exclude internal maintenance notes and historical experiment dumps unless a
  compact artifact is required to reproduce a reported result.

## Bug Review and Validation

Review the imported changes before treating them as publishable. The minimum
verification set is:

1. Compile every changed Python module.
2. Run shell syntax checks for every public shell entrypoint.
3. Run existing Sparse-vLLM unit tests affected by the FutureKV integration.
4. Run FutureKV cache-manager, judge, scheduler, sampler, kernel-contract, and
   MathVision-driver tests.
5. Run standalone training tests for data validation, checkpoint export, and
   resume behavior.
6. Run a cross-package checkpoint contract test.
7. Run a CPU/lightweight training smoke test where feasible.
8. Run a minimal GPU FutureKV inference smoke test when the environment and
   model checkpoint are available.
9. Compare a small deterministic full-KV baseline and FutureKV run for output
   completeness, sample accounting, and reproducibility metadata.

Any discovered bug is first reproduced with a focused failing test. Fixes stay
in the new repository unless the user separately requests backporting to the
active source checkout.

## Completion Criteria

The repository is ready when:

- It installs as an inference package from the root.
- The training subsystem installs independently.
- Training produces a checkpoint accepted by the inference loader.
- FutureKV inference and MathVision evaluation have documented commands.
- Relevant automated tests pass.
- No required code or configuration depends on external source checkouts.
- Repository hygiene and secret/path scans pass.
- README, licensing, attribution, configuration, and reproduction instructions
  are sufficient for a new user to run the workflow with their own model and
  dataset paths.
