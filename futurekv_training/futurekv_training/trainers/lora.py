"""Small dependency-free LoRA layers used by the optional SFT entrypoint."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if int(rank) <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.lora_A = nn.Linear(
            base_layer.in_features,
            self.rank,
            bias=False,
            device=base_layer.weight.device,
            dtype=torch.float32,
        )
        self.lora_B = nn.Linear(
            self.rank,
            base_layer.out_features,
            bias=False,
            device=base_layer.weight.device,
            dtype=torch.float32,
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(values)
        delta = self.lora_B(
            self.lora_A(self.dropout(values).to(self.lora_A.weight.dtype))
        )
        return base + delta.to(base.dtype) * self.scaling


def inject_lora(
    model: nn.Module,
    *,
    target_suffixes: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[str]:
    suffixes = tuple(str(suffix) for suffix in target_suffixes)
    replaced = []
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in suffixes):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._modules[child_name] = LoRALinear(
            module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        replaced.append(name)
    if not replaced:
        raise ValueError(f"No Linear modules matched LoRA targets: {suffixes}")
    return replaced


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_A." in name or ".lora_B." in name
    }
