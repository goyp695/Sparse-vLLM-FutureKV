# Sync FutureKV Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已验证的 FutureKV 物理 slot repack 修复同步到公开仓库，并逐项评估 scheduler、ModelRunner、Sampler 修复是否适配公开版。

**Architecture:** 保留公开仓库当前的 cache-manager-first 架构。物理 repack 在 FutureKV cache manager 内完成：先按 head gather K/V，再写入一组共享目标 slot，并释放尾部 slot。scheduler、ModelRunner、Sampler 不整文件覆盖，只根据公开版已有接口逐项移植并配套回归测试。

**Tech Stack:** Python、PyTorch、pytest、Triton（仅在已有 GPU kernel 测试中验证）。

---

### Task 1: 同步 FutureKV physical slot repack

**Files:**
- Modify: `src/sparsevllm/engine/cache_manager/futurekv.py`
- Modify: `src/sparsevllm/engine/sparse_controller.py`
- Modify: `tests/test_futurekv_cache_manager.py`

- [ ] **Step 1: 为不同 head 的选择写失败测试**

测试必须构造两个 head 的不相交选择，断言压缩后的物理 row 恰好等于 target length，并检查 K/V 数值被正确搬运。

- [ ] **Step 2: 运行测试确认当前 union-retention 行为失败**

```bash
PYTHONPATH=src conda run -n vllm python -m pytest -q tests/test_futurekv_cache_manager.py
```

预期失败：当前实现使用 head 选择结果的物理 slot 并集，`row_seq_lens` 大于 target length。

- [ ] **Step 3: 为 `apply_futurekv_head_keep` 增加 gathered K/V 契约**

接口增加：

```python
*, gathered_key: torch.Tensor, gathered_value: torch.Tensor
```

实现顺序固定为：

1. 校验 tensor rank、head 数、长度、dtype、device、slot 唯一性和 keep 范围。
2. 在任何 cache 写入前 gather selected K/V。
3. 将旧 row 的前 `target_len` 个 slot 作为目标 slot。
4. 用 `index_copy_` 写入 selected K/V。
5. 两个 head 使用同一组目标 slot，但保留各自 logical indices。
6. 只释放旧 row 的尾部 slot，并把 row length 设置为 target length。

- [ ] **Step 4: 从 SparseController 传递 gathered K/V**

在计算 keep indices 时复用已经 gather 的 `key/value`，调用：

```python
self.cache_manager.apply_futurekv_head_keep(
    layer_idx,
    seq,
    head_slots,
    head_indices,
    keep_indices,
    gathered_key=key,
    gathered_value=value,
)
```

- [ ] **Step 5: 运行 FutureKV cache manager 与 checkpoint 相关测试**

```bash
PYTHONPATH=src:futurekv_training conda run -n vllm python -m pytest -q \
  tests/test_futurekv_cache_manager.py \
  tests/test_futurekv_attention_fail_fast.py \
  tests/test_futurekv_checkpoint_contract.py \
  tests/test_futurekv_judge.py
```

- [ ] **Step 6: 提交独立 commit**

```bash
git add src/sparsevllm/engine/cache_manager/futurekv.py \
  src/sparsevllm/engine/sparse_controller.py \
  tests/test_futurekv_cache_manager.py
git commit -m "fix: repack FutureKV physical slots"
```

### Task 2: 逐项验证 scheduler 修复

**Files:**
- Inspect/modify: `src/sparsevllm/engine/scheduler.py`
- Inspect/modify: `src/sparsevllm/engine/cache_manager/base.py`
- Test: `tests/test_prefill_schedule_policy.py`

- [ ] **Step 1: 对照公开版接口与源任务接口**

只比较 prompt admission、暂时 defer、永久不可容纳 fail-fast 三类逻辑；忽略 DeltaKV 专属调度改动。

- [ ] **Step 2: 为公开版缺失行为写最小失败测试**

覆盖：

```text
物理空间暂时不足 -> 保持 waiting/defer
单个 prompt 超过总容量 -> 明确 RuntimeError
已有 completion 的 decode request 不被错误重放
```

- [ ] **Step 3: 运行失败测试并确认是公开版缺少行为**

```bash
PYTHONPATH=src conda run -n vllm python -m pytest -q \
  tests/test_prefill_schedule_policy.py
```

- [ ] **Step 4: 只移植与公开版 memory-oracle 接口匹配的最小改动**

不复制源任务的 DeltaKV 分支、preemption 重构或 config 字段。

- [ ] **Step 5: 运行完整 scheduler 回归**

```bash
PYTHONPATH=src conda run -n vllm python -m pytest -q \
  tests/test_prefill_schedule_policy.py \
  tests/test_tp_rpc.py \
  tests/test_engine_shutdown.py
```

- [ ] **Step 6: 若有实际改动则单独提交，否则记录“不需同步”**

### Task 3: 逐项验证 ModelRunner chunked-prefill 修复

**Files:**
- Inspect/modify: `src/sparsevllm/engine/model_runner.py`
- Inspect/modify: `src/sparsevllm/engine/sequence.py`
- Test: add or modify the smallest relevant runtime test

- [ ] **Step 1: 检查公开版是否存在中间 prefill token 采样历史**

只有公开版确实维护中间 chunk 的输出 token 或 RNG 状态时，才移植该修复。

- [ ] **Step 2: 若存在，写测试证明 chunk size 不应改变首个真实生成 token**

测试必须区分“中间 chunk 计算出的临时 logits”与“最后 chunk 才提交的生成 token”。

- [ ] **Step 3: 运行失败测试**

```bash
PYTHONPATH=src conda run -n vllm python -m pytest -q \
  tests/test_sampler.py tests/test_prefill_schedule_policy.py
```

- [ ] **Step 4: 只移植公开版所需的提交时机修复**

不复制源任务的完整 ModelRunner 重构。

- [ ] **Step 5: 运行 runtime 回归并单独提交或记录不适用**

### Task 4: 逐项验证 Sampler 修复

**Files:**
- Inspect/modify: `src/sparsevllm/layers/sampler.py`
- Inspect/modify: `tests/test_sampler.py`

- [ ] **Step 1: 对照采样器输入输出契约**

确认公开版是否需要请求级 RNG、presence/repetition penalty 历史或仅使用当前 logits。

- [ ] **Step 2: 为公开版真实缺陷写最小失败测试**

只覆盖公开版实际暴露的行为，不引入源任务不存在的字段。

- [ ] **Step 3: 运行失败测试后写最小修复**

- [ ] **Step 4: 运行 sampler、Qwen3-VL driver 和 FutureKV 端到端回归**

```bash
PYTHONPATH=src:futurekv_training conda run -n vllm python -m pytest -q \
  tests/test_sampler.py \
  tests/test_qwen3vl_mathvision_driver.py \
  tests/test_futurekv_attention_fail_fast.py \
  futurekv_training/tests
```

- [ ] **Step 5: 单独提交或记录不适用**

### Task 5: 最终验证与 GitHub 同步准备

- [ ] **Step 1: 运行公开仓库卫生、编译和 diff 检查**

```bash
git diff --check
PYTHONPATH=src:futurekv_training conda run -n vllm python -m compileall -q src benchmark futurekv_training/futurekv_training
conda run -n vllm python scripts/validation/check_public_repo.py
```

- [ ] **Step 2: 运行完整相关测试并记录结果**

- [ ] **Step 3: 检查工作区只包含已验证提交**

- [ ] **Step 4: 推送 `main` 并核对 GitHub commit**

