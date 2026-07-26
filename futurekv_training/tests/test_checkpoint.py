import json
from pathlib import Path

import pytest
import torch
from torch import nn

from futurekv_training.checkpoint import (
    export_judge_checkpoint,
    validate_checkpoint,
)


class TinyJudgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.judge_model = nn.Module()
            layer.self_attn.judge_model.fc1 = HeadWiseLinear(1, 16, 4)
            layer.self_attn.judge_model.fc2 = HeadWiseLinear(1, 4, 1)
            self.layers.append(layer)


class HeadWiseLinear(nn.Module):
    def __init__(self, heads: int, input_size: int, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(heads, input_size, output_size))
        self.bias = nn.Parameter(torch.randn(heads, output_size))


def metadata():
    return {
        "base_model": "example/Qwen3-VL",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_layers": 2,
        "num_kv_heads": 1,
        "dtype": "float32",
        "training_method": "judge_only",
        "futurekv": {
            "budget": 1024,
            "window_size": 1024,
            "step_drop": 256,
            "divide_length": 128,
        },
        "creation": {"seed": 42},
    }


def test_export_and_validate_checkpoint_pair(tmp_path: Path):
    weights_path, metadata_path = export_judge_checkpoint(
        TinyJudgeModel(), tmp_path, metadata()
    )

    assert weights_path.name == "judge_model.pt"
    assert metadata_path.name == "judge_model_meta.json"
    parsed = validate_checkpoint(tmp_path)
    assert parsed.schema_version == 1
    assert parsed.num_layers == 2
    assert parsed.num_tensors == 8
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    assert all(".judge_model." in key for key in state)
    assert json.loads(metadata_path.read_text())["schema_version"] == 1


def test_validate_rejects_missing_tensor(tmp_path: Path):
    export_judge_checkpoint(TinyJudgeModel(), tmp_path, metadata())
    weights_path = tmp_path / "judge_model.pt"
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    state.pop(next(key for key in state if key.endswith("fc2.bias")))
    torch.save(state, weights_path)

    with pytest.raises(ValueError, match="missing tensors"):
        validate_checkpoint(tmp_path)


def test_validate_rejects_head_shape_mismatch(tmp_path: Path):
    export_judge_checkpoint(TinyJudgeModel(), tmp_path, metadata())
    metadata_path = tmp_path / "judge_model_meta.json"
    payload = json.loads(metadata_path.read_text())
    payload["num_kv_heads"] = 2
    metadata_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="num_kv_heads"):
        validate_checkpoint(tmp_path)
