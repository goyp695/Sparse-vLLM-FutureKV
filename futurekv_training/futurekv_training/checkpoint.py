"""Versioned checkpoint contract shared with FutureKV inference by files only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import torch
from torch import nn


SCHEMA_VERSION = 1
WEIGHTS_NAME = "judge_model.pt"
METADATA_NAME = "judge_model_meta.json"
_KEY_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.self_attn\.judge_model\."
    r"(?P<name>fc[12]\.(?:weight|bias))$"
)
_TENSOR_NAMES = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}


@dataclass(frozen=True)
class CheckpointMetadata:
    schema_version: int
    base_model: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_kv_heads: int
    dtype: str
    training_method: str
    futurekv: dict[str, Any]
    creation: dict[str, Any]
    num_tensors: int


def _resolve_paths(path: str | Path) -> tuple[Path, Path]:
    path = Path(path)
    directory = path if path.is_dir() or path.suffix == "" else path.parent
    weights_path = path if path.name == WEIGHTS_NAME else directory / WEIGHTS_NAME
    return weights_path, directory / METADATA_NAME


def _normalized_judge_state(
    state: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        match = _KEY_RE.search(key)
        if match is None or not torch.is_tensor(value):
            continue
        normalized[
            f"model.language_model.layers.{int(match.group('layer'))}."
            f"self_attn.judge_model.{match.group('name')}"
        ] = value.detach().cpu().contiguous()
    if not normalized:
        raise ValueError("No FutureKV judge_model tensors were found.")
    return normalized


def _validate_state(
    state: Mapping[str, torch.Tensor],
    *,
    num_layers: int,
    num_kv_heads: int,
) -> None:
    layers: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        match = _KEY_RE.search(key)
        if match is not None:
            layers.setdefault(int(match.group("layer")), {})[match.group("name")] = tensor
    expected_layers = set(range(int(num_layers)))
    if set(layers) != expected_layers:
        raise ValueError(
            "Judge checkpoint layers do not match num_layers: "
            f"found={sorted(layers)} expected={sorted(expected_layers)}."
        )
    for layer_idx, tensors in layers.items():
        missing = _TENSOR_NAMES.difference(tensors)
        if missing:
            raise ValueError(
                f"Judge layer {layer_idx} is missing tensors: {sorted(missing)}."
            )
        if any(int(tensor.shape[0]) != int(num_kv_heads) for tensor in tensors.values()):
            raise ValueError(
                f"Judge layer {layer_idx} tensor num_kv_heads does not match "
                f"metadata num_kv_heads={num_kv_heads}."
            )
        fc1w, fc1b = tensors["fc1.weight"], tensors["fc1.bias"]
        fc2w, fc2b = tensors["fc2.weight"], tensors["fc2.bias"]
        if fc1w.ndim != 3 or fc1b.ndim != 2 or fc2w.ndim != 3 or fc2b.ndim != 2:
            raise ValueError(f"Judge layer {layer_idx} has invalid tensor ranks.")
        judge_hidden = int(fc1w.shape[2])
        if tuple(fc1b.shape[1:]) != (judge_hidden,):
            raise ValueError(f"Judge layer {layer_idx} fc1 bias shape mismatch.")
        if tuple(fc2w.shape[1:]) != (judge_hidden, 1):
            raise ValueError(f"Judge layer {layer_idx} fc2 weight shape mismatch.")
        if tuple(fc2b.shape[1:]) != (1,):
            raise ValueError(f"Judge layer {layer_idx} fc2 bias shape mismatch.")


def _metadata_from_payload(payload: Mapping[str, Any]) -> CheckpointMetadata:
    required = {
        "schema_version",
        "base_model",
        "hidden_size",
        "intermediate_size",
        "num_layers",
        "num_kv_heads",
        "dtype",
        "training_method",
        "futurekv",
        "creation",
        "num_tensors",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Checkpoint metadata is missing fields: {sorted(missing)}.")
    if int(payload["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema_version={payload['schema_version']}; "
            f"expected {SCHEMA_VERSION}."
        )
    return CheckpointMetadata(
        schema_version=SCHEMA_VERSION,
        base_model=str(payload["base_model"]),
        hidden_size=int(payload["hidden_size"]),
        intermediate_size=int(payload["intermediate_size"]),
        num_layers=int(payload["num_layers"]),
        num_kv_heads=int(payload["num_kv_heads"]),
        dtype=str(payload["dtype"]),
        training_method=str(payload["training_method"]),
        futurekv=dict(payload["futurekv"]),
        creation=dict(payload["creation"]),
        num_tensors=int(payload["num_tensors"]),
    )


def validate_checkpoint(path: str | Path) -> CheckpointMetadata:
    weights_path, metadata_path = _resolve_paths(path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing FutureKV weights: {weights_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing FutureKV metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = _metadata_from_payload(json.load(handle))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("FutureKV weights must contain a tensor mapping.")
    normalized = _normalized_judge_state(state)
    _validate_state(
        normalized,
        num_layers=metadata.num_layers,
        num_kv_heads=metadata.num_kv_heads,
    )
    if len(normalized) != metadata.num_tensors:
        raise ValueError(
            f"Checkpoint num_tensors mismatch: metadata={metadata.num_tensors}, "
            f"weights={len(normalized)}."
        )
    return metadata


def _atomic_torch_save(state: Mapping[str, torch.Tensor], destination: Path) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(state), temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json_save(payload: Mapping[str, Any], destination: Path) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def export_judge_checkpoint(
    model: nn.Module,
    output_dir: str | Path,
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = _normalized_judge_state(model.state_dict())
    payload = dict(metadata)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload["num_tensors"] = len(state)
    creation = dict(payload.get("creation", {}))
    creation.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    payload["creation"] = creation
    parsed = _metadata_from_payload(payload)
    _validate_state(
        state,
        num_layers=parsed.num_layers,
        num_kv_heads=parsed.num_kv_heads,
    )

    weights_path = output_dir / WEIGHTS_NAME
    metadata_path = output_dir / METADATA_NAME
    _atomic_torch_save(state, weights_path)
    _atomic_json_save(asdict(parsed), metadata_path)
    validate_checkpoint(output_dir)
    return weights_path, metadata_path


def load_judge_checkpoint(model: nn.Module, path: str | Path) -> CheckpointMetadata:
    """Load a validated public checkpoint into a training-side judge container."""
    metadata = validate_checkpoint(path)
    weights_path, _ = _resolve_paths(path)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    normalized = _normalized_judge_state(state)
    training_state = {}
    for key, tensor in normalized.items():
        marker = "model.language_model."
        training_state[key[len(marker):] if key.startswith(marker) else key] = tensor
    incompatible = model.load_state_dict(training_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Judge checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}."
        )
    return metadata
