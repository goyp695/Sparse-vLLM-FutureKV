import torch
from torch import nn

from futurekv_training.trainers.lora import inject_lora, lora_state_dict


def test_lora_injection_freezes_base_and_exports_only_adapter():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    replaced = inject_lora(model, target_suffixes=("0",), rank=2, alpha=4)
    assert replaced == ["0"]
    assert not model[0].base_layer.weight.requires_grad
    output = model(torch.randn(3, 4)).sum()
    output.backward()
    state = lora_state_dict(model)
    assert sorted(state) == ["0.lora_A.weight", "0.lora_B.weight"]
