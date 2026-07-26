"""Validated MathVision-style multimodal training records."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MultimodalRecord:
    sample_id: str
    messages: tuple[dict[str, Any], ...]
    image_paths: tuple[Path, ...]


def load_multimodal_records(
    dataset_path: str | Path,
    *,
    image_root: str | Path,
) -> list[MultimodalRecord]:
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Training dataset does not exist: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Training dataset must be a non-empty JSON list.")

    root = Path(image_root).expanduser().resolve()
    records: list[MultimodalRecord] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Training record {index} must be an object.")
        messages = item.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Training record {index} requires user and assistant messages.")
        if not all(isinstance(message, dict) and "content" in message for message in messages):
            raise ValueError(f"Training record {index} contains an invalid message.")
        sample_id = str(item.get("id", index))
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate training sample id: {sample_id}")
        seen_ids.add(sample_id)
        paths = []
        for raw_path in item.get("images") or []:
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Training image does not exist: {path}")
            paths.append(path)
        records.append(
            MultimodalRecord(sample_id, tuple(messages), tuple(paths))
        )
    return records
