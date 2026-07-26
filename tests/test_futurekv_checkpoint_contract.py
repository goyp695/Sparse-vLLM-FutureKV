import json
from pathlib import Path

import pytest
import torch

from sparsevllm.engine.futurekv_judge import load_futurekv_judge_state


def _state():
    prefix = "model.language_model.layers.0.self_attn.judge_model"
    return {
        f"{prefix}.fc1.weight": torch.randn(2, 16, 4),
        f"{prefix}.fc1.bias": torch.randn(2, 4),
        f"{prefix}.fc2.weight": torch.randn(2, 4, 1),
        f"{prefix}.fc2.bias": torch.randn(2, 1),
    }


def _metadata():
    return {
        "schema_version": 1,
        "base_model": "example/Qwen3-VL",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_layers": 1,
        "num_kv_heads": 2,
        "dtype": "float32",
        "training_method": "judge_only",
        "futurekv": {
            "budget": 1024,
            "window_size": 1024,
            "step_drop": 256,
            "divide_length": 128,
        },
        "creation": {"seed": 42},
        "num_tensors": 4,
    }


def test_runtime_loads_checkpoint_directory_or_file(tmp_path: Path):
    weights_path = tmp_path / "judge_model.pt"
    torch.save(_state(), weights_path)
    (tmp_path / "judge_model_meta.json").write_text(json.dumps(_metadata()))

    assert sorted(load_futurekv_judge_state(tmp_path)) == [0]
    assert sorted(load_futurekv_judge_state(weights_path)) == [0]


def test_runtime_rejects_missing_metadata_for_public_checkpoint(tmp_path: Path):
    torch.save(_state(), tmp_path / "judge_model.pt")

    with pytest.raises(ValueError, match="metadata"):
        load_futurekv_judge_state(tmp_path)


def test_runtime_rejects_incompatible_schema(tmp_path: Path):
    torch.save(_state(), tmp_path / "judge_model.pt")
    metadata = _metadata()
    metadata["schema_version"] = 999
    (tmp_path / "judge_model_meta.json").write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="schema_version"):
        load_futurekv_judge_state(tmp_path)
