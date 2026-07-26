# FutureKV Public Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an independently publishable Sparse-vLLM repository that can train a FutureKV judge, export a versioned checkpoint, load it in native Sparse-vLLM inference, and evaluate Qwen3-VL on MathVision.

**Architecture:** Keep FutureKV inference inside the existing cache-manager-first Sparse-vLLM runtime. Keep training in the independently installable `futurekv_training` project, with no imports across the package boundary; exchange only `judge_model.pt` and `judge_model_meta.json`.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, Sparse-vLLM, pytest, setuptools, Bash.

## Global Constraints

- Preserve Sparse-vLLM history, Apache-2.0 licensing, and inherited attribution.
- Keep the historical inference default `futurekv_window_size=1024`, `futurekv_budget=1024`, and `futurekv_step_drop=256`.
- The root package and `futurekv_training` must install independently.
- Runtime code must not import `futurekv_training`; training code must not import `sparsevllm`.
- Missing or incompatible checkpoints, datasets, models, and evaluation resources fail before GPU work starts.
- Do not copy logs, generated results, datasets, weights, adapters, caches, environments, absolute server paths, or credentials.
- Every bug fix starts with a focused failing test.

---

### Task 1: Import and validate the native FutureKV runtime

**Files:**
- Modify: `src/sparsevllm/config.py`
- Modify: `src/sparsevllm/method_registry.py`
- Modify: `src/sparsevllm/engine/cache_manager/base.py`
- Modify: `src/sparsevllm/engine/cache_manager/__init__.py`
- Create: `src/sparsevllm/engine/cache_manager/futurekv.py`
- Create: `src/sparsevllm/engine/futurekv_judge.py`
- Modify: `src/sparsevllm/engine/sparse_controller.py`
- Modify: `src/sparsevllm/layers/attention.py`
- Modify: the active runtime support files listed by `git status` in the read-only source checkout
- Test: `tests/test_futurekv_cache_manager.py`
- Test: `tests/test_futurekv_judge.py`
- Test: `tests/test_futurekv_attention_fail_fast.py`
- Test: `tests/test_vllm_futurekv_head_slots.py`
- Test: `tests/test_prefill_schedule_policy.py`

**Interfaces:**
- Consumes: Sparse-vLLM `Config`, `CacheManager.create`, generic attention cache hooks, and tensor-parallel rank metadata.
- Produces: first-class method name `futurekv`, `FutureKVCacheManager`, `load_futurekv_judge_state(path)`, and validated FutureKV configuration fields.

- [ ] Copy only the FutureKV-focused tests from the active read-only checkout.
- [ ] Run the focused tests and confirm they fail because FutureKV runtime symbols are absent.
- [ ] Apply the active checkout's tracked runtime diff and copy its untracked FutureKV runtime modules.
- [ ] Review every imported hunk; remove unrelated experimental changes and machine-specific assumptions.
- [ ] Run the focused tests and compile all imported Python modules.
- [ ] Commit with `feat: add FutureKV runtime`.

### Task 2: Add Qwen3-VL and MathVision inference

**Files:**
- Create: `src/sparsevllm/models/qwen3_vl.py`
- Modify: `src/sparsevllm/utils/loader.py`
- Modify: `benchmark/multimodal/model_adapters/qwen3_vl.py`
- Create: `benchmark/multimodal/image_qa/mathvision_qwen3vl.py`
- Create: `configs/futurekv/qwen3vl_mathvision.json`
- Create: `scripts/futurekv/run_mathvision.sh`
- Test: `tests/test_qwen3vl_mathvision_driver.py`

**Interfaces:**
- Consumes: `sparsevllm.LLM`, `SamplingParams`, FutureKV root config fields, a JSON MathVision dataset, and local image paths.
- Produces: a CLI with explicit model, dataset, image root, output directory, checkpoint, seed, budget, drop step, window size, tensor parallel size, and decoding parameters.

- [ ] Copy the driver test and verify it fails on the clean repository.
- [ ] Import the Qwen3-VL model, adapter, loader registration, and MathVision driver from the active checkout.
- [ ] Add a relative-path JSON example whose FutureKV defaults are `budget=1024`, `window_size=1024`, and `step_drop=256`.
- [ ] Add a shell entrypoint that rejects missing model, dataset, image root, checkpoint, and output options before launching Python.
- [ ] Run the driver test, shell `bash -n`, and Python compilation.
- [ ] Commit with `feat: add FutureKV MathVision runner`.

### Task 3: Define the cross-package checkpoint contract

**Files:**
- Create: `futurekv_training/futurekv_training/checkpoint.py`
- Create: `futurekv_training/tests/test_checkpoint.py`
- Create: `tests/test_futurekv_checkpoint_contract.py`
- Modify: `src/sparsevllm/engine/futurekv_judge.py`

**Interfaces:**
- Produces: `export_judge_checkpoint(model, output_dir, metadata) -> tuple[Path, Path]`, `validate_checkpoint(path) -> CheckpointMetadata`, and runtime loading from either a `judge_model.pt` file or its containing directory.
- Artifact schema: `schema_version=1`; metadata includes `base_model`, `hidden_size`, `intermediate_size`, `num_layers`, `num_kv_heads`, `dtype`, `training_method`, `futurekv`, and `creation`.

- [ ] Write tests that create a two-layer synthetic judge state, validate metadata, reject missing tensors or shape mismatches, and check directory/file loading.
- [ ] Run both tests and confirm failure because the contract implementation is absent.
- [ ] Implement strict CPU loading, normalized tensor keys, tensor-parallel head slicing validation, and atomic temporary-file replacement.
- [ ] Run the cross-package and runtime judge tests.
- [ ] Commit with `feat: add FutureKV checkpoint contract`.

### Task 4: Create the independent judge-training package

**Files:**
- Create: `futurekv_training/pyproject.toml`
- Create: `futurekv_training/README.md`
- Create: `futurekv_training/futurekv_training/__init__.py`
- Create: `futurekv_training/futurekv_training/modeling.py`
- Create: `futurekv_training/futurekv_training/cache.py`
- Create: `futurekv_training/futurekv_training/data.py`
- Create: `futurekv_training/futurekv_training/cli/train_judge.py`
- Create: `futurekv_training/configs/smoke.json`
- Create: `futurekv_training/tests/test_data.py`
- Create: `futurekv_training/tests/test_modeling.py`
- Create: `futurekv_training/tests/test_package_boundaries.py`

**Interfaces:**
- Consumes: explicit base-model and dataset paths plus checked JSON configuration.
- Produces: `SelectionModel`, validated multimodal records, oracle-label construction, judge-only optimization, and the Task 3 checkpoint pair.

- [ ] Write tests for required dataset fields, missing images, empty datasets, selection-model tensor shapes, and forbidden cross-package imports.
- [ ] Run tests and confirm failure because the standalone package is absent.
- [ ] Extract only judge architecture, cache/oracle logic, and dataset behavior from `goyp/sft_lora`; remove `GOYP_ROOT`, hard-coded devices, and implicit local paths.
- [ ] Add explicit bounded step counts, deterministic seed handling, and output metadata.
- [ ] Run package tests and a synthetic CPU forward/backward smoke test.
- [ ] Commit with `feat: add standalone judge training`.

### Task 5: Port two-stage training and optional LoRA

**Files:**
- Create: `futurekv_training/futurekv_training/trainers/two_stage.py`
- Create: `futurekv_training/futurekv_training/trainers/lora.py`
- Create: `futurekv_training/futurekv_training/cli/train_two_stage.py`
- Create: `futurekv_training/futurekv_training/cli/train_lora.py`
- Create: `futurekv_training/scripts/run_two_stage.sh`
- Create: `futurekv_training/scripts/run_lora.sh`
- Create: `futurekv_training/tests/test_two_stage.py`
- Create: `futurekv_training/tests/test_resume.py`

**Interfaces:**
- Consumes: a validated stage-one checkpoint, fresh raw generation JSONL, explicit output paths, and bounded training/generation limits.
- Produces: separate raw, filtered, per-sample, aggregate, debug, and checkpoint artifacts; resume skips only completed sample IDs and records skip reasons.

- [ ] Write tests for correct-answer filtering, duplicate IDs, repeat-like tails, bounded attempts, cut-position validation, and resume accounting.
- [ ] Run the tests and confirm the behaviors are absent.
- [ ] Extract the current two-stage and LoRA workflows without copying datasets, models, outputs, or local path assertions.
- [ ] Make all GPU/device choices explicit CLI options and fail when requested devices are unavailable.
- [ ] Run unit tests plus shell syntax checks.
- [ ] Commit with `feat: add two-stage FutureKV training`.

### Task 6: Public documentation and repository hygiene

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`
- Create: `docs/features/futurekv.md`
- Create: `docs/getting_started/futurekv-training.md`
- Create: `docs/getting_started/futurekv-inference.md`
- Create: `docs/getting_started/futurekv-reproducibility.md`
- Create: `scripts/validation/check_public_repo.py`
- Create: `tests/test_public_repo_hygiene.py`

**Interfaces:**
- Produces: install, train, validate, infer, evaluate, and reproduction instructions using placeholders or relative paths.

- [ ] Write a hygiene test that rejects tracked caches, logs, results, weights, datasets, PID files, `/home/`, `/media/`, private-key headers, and common API-token assignments.
- [ ] Run it and confirm the current inherited repository violations are reported.
- [ ] Extend ignores, clean tracked public documentation/configuration references, and add provenance and Apache-2.0 attribution.
- [ ] Document `window_size=1024` as the reproduced Sparse-vLLM default and distinguish it from `step_drop=256`.
- [ ] Run the hygiene test and scan all tracked files.
- [ ] Commit with `docs: document FutureKV workflow`.

### Task 7: End-to-end verification and bug review

**Files:**
- Modify: only files with failures reproduced by focused tests.

**Interfaces:**
- Produces: installable inference and training packages plus a verified checkpoint handoff.

- [ ] Check GPU availability before GPU tasks; use an idle device or report that GPU smoke tests are deferred.
- [ ] Run `python -m compileall` on root FutureKV and training modules.
- [ ] Run `bash -n` for every public shell entrypoint.
- [ ] Build wheels for the root and training projects independently and inspect their contents.
- [ ] Run all FutureKV, Qwen3-VL, checkpoint-contract, training, and hygiene tests.
- [ ] Run the repository-local code-review skill against the full branch diff; reproduce each actionable bug with a failing test before fixing.
- [ ] Run a CPU synthetic checkpoint export/load smoke test.
- [ ] If an idle GPU and local model/data/checkpoint exist, run a one-sample FutureKV inference smoke test; otherwise record the exact missing prerequisite.
- [ ] Commit verified fixes with conventional commit messages.
