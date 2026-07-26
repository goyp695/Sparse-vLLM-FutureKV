"""Deterministic filtering and resume utilities for two-stage judge training."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable


_BOXED = re.compile(r"\\boxed\s*(?:\{([^{}]+)\}|([^\s,.;]+))")


def extract_boxed_answer(text: str) -> str | None:
    matches = list(_BOXED.finditer(str(text)))
    if not matches:
        return None
    return (matches[-1].group(1) or matches[-1].group(2)).strip()


def _normalize_answer(value: Any) -> str:
    value = str(value).strip()
    boxed = extract_boxed_answer(value)
    if boxed is not None:
        value = boxed
    return re.sub(r"\s+", "", value).replace("$", "").lower()


def has_repeated_tail(text: str, *, min_repeats: int = 4) -> bool:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if len(normalized) < 32:
        return False
    for width in range(4, min(128, len(normalized) // min_repeats) + 1):
        tail = normalized[-width:]
        if normalized.endswith(tail * min_repeats):
            return True
    words = normalized.split()
    if len(words) >= min_repeats * 2:
        for width in range(1, min(16, len(words) // min_repeats) + 1):
            unit = words[-width:]
            if words[-width * min_repeats:] == unit * min_repeats:
                return True
    return False


def filter_correct_generations(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted = []
    skipped: Counter[str] = Counter()
    seen: set[str] = set()
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_uid", row.get("id", index)))
        if sample_id in seen:
            raise ValueError(f"Duplicate generated sample id: {sample_id}")
        seen.add(sample_id)
        if row.get("status", "success") != "success":
            skipped["generation_error"] += 1
            continue
        response = str(row.get("response", row.get("pred", "")))
        if not response:
            skipped["empty_response"] += 1
            continue
        if has_repeated_tail(response):
            skipped["repeated_tail"] += 1
            continue
        expected = row.get("answer", row.get("ground_truth", row.get("gt_answer")))
        predicted = extract_boxed_answer(response)
        if expected is None or predicted is None:
            skipped["parse_failed"] += 1
            continue
        if _normalize_answer(predicted) != _normalize_answer(expected):
            skipped["incorrect"] += 1
            continue
        accepted.append(dict(row))
    return accepted, dict(sorted(skipped.items()))


def choose_cut_position(
    token_count: int,
    *,
    min_tokens: int,
    future_tokens: int,
    seed: int,
) -> int:
    lower = int(min_tokens)
    upper = int(token_count) - int(future_tokens)
    if lower > upper:
        raise ValueError(
            f"No valid cut position: token_count={token_count}, "
            f"min_tokens={min_tokens}, future_tokens={future_tokens}."
        )
    return random.Random(int(seed)).randint(lower, upper)


def completed_sample_ids(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.is_file():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status", "success") == "success":
                completed.add(str(row.get("sample_uid", row.get("id"))))
    completed.discard("None")
    return completed
