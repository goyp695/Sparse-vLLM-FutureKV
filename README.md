# FutureKV

FutureKV 是面向 Qwen3-VL 长上下文推理的按 KV head 稀疏缓存方案。本仓库
包含原生推理实现、独立的 Judge 训练代码，以及 MathVision 评测入口。
训练包和推理包可以分别安装，二者通过版本化的 Judge checkpoint 对接。

## 仓库内容

```text
src/sparsevllm/        原生 Sparse-vLLM runtime 与 FutureKV cache manager
benchmark/multimodal/ Qwen3-VL MathVision 推理驱动
futurekv_training/     独立的 Judge、二阶段续训和可选 LoRA 训练包
scripts/futurekv/      可直接运行的评测脚本
docs/                  设计、训练、推理和复现实验说明
tests/                 runtime、kernel、checkpoint 与驱动测试
```

## FutureKV 工作方式

每个 KV head 独立维护稀疏缓存，最近 token 受到保护，历史 token 由训练得到的
Judge 根据 key/value 特征和 future-attention 信息打分。推理时使用固定的
checkpoint pair：

```text
judge_model.pt
judge_model_meta.json
```

默认参数与含义：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `kv_budget` | `1024` | 每个 KV head 最多保留的 KV token 数 |
| `window_size` | `1024` | Judge 特征使用的 query 环形缓存长度，不是额外 KV budget |
| `step_drop` | `256` | 每次淘汰决策处理的 token 数 |
| `divide_length` | `128` | Judge/runtime 的决策分段长度 |

## 安装

推理 runtime：

```bash
pip install -e .
```

独立训练包：

```bash
pip install -e './futurekv_training[test]'
```

建议使用与本地 Qwen3-VL 模型匹配的 Transformers、PyTorch 和 CUDA/Triton
版本；完整依赖与环境说明见 [Getting Started](docs/getting_started/README.md)。

## Judge 训练

训练数据是 JSON list。每条记录包含 `messages`、可选唯一 `id` 和相对于
`--image-root` 的 `images` 路径：

```bash
futurekv-train-judge \
  --base-model /path/to/Qwen3-VL \
  --dataset /path/to/train.json \
  --image-root /path/to/images \
  --output-dir ./outputs/judge \
  --device cuda:0 \
  --max-steps 1000
```

训练会冻结 Qwen3-VL，仅更新每层的 Judge，并原子写出 checkpoint pair。
二阶段正确性过滤/续训和可选 LoRA 训练见
[FutureKV training](docs/getting_started/futurekv-training.md)。

## Qwen3-VL 推理

```bash
scripts/futurekv/run_mathvision.sh \
  /path/to/Qwen3-VL \
  /path/to/mathvision.json \
  /path/to/images \
  /path/to/judge/judge_model.pt \
  ./outputs/mathvision_futurekv.jsonl
```

脚本会检查模型、数据、图片目录和 checkpoint；`judge_model_meta.json` 必须
与 `judge_model.pt` 位于同一目录。直接使用 Python 时选择
`--backend sparsevllm --method futurekv`。推理入口使用 greedy 解码
（`temperature=0`）以保证评测可复现。更多说明见
[FutureKV inference](docs/getting_started/futurekv-inference.md)。

## 验证

运行 FutureKV 与训练包测试：

```bash
PYTHONPATH=src:futurekv_training pytest -q \
  tests/test_futurekv_attention_fail_fast.py \
  tests/test_futurekv_cache_manager.py \
  tests/test_futurekv_checkpoint_contract.py \
  tests/test_futurekv_judge.py \
  tests/test_qwen3vl_mathvision_driver.py \
  futurekv_training/tests
```

仓库还提供 GPU kernel 数值校验、公开仓库卫生检查和 wheel 构建检查。

## 许可

本项目使用 [Apache License 2.0](LICENSE)。
