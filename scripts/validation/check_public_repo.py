#!/usr/bin/env python3
"""Reject machine-local or generated artifacts from FutureKV public surfaces."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess


PUBLIC_PREFIXES = (
    "src/sparsevllm/",
    "benchmark/multimodal/",
    "configs/futurekv/",
    "scripts/futurekv/",
    "futurekv_training/",
    "docs/features/futurekv.md",
    "docs/getting_started/futurekv-",
)
PUBLIC_ROOT_FILES = {"README.md", ".gitignore", "pyproject.toml"}
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".log",
    ".pid",
    ".jsonl",
    ".rej",
    ".orig",
}
FORBIDDEN_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "wandb",
    "logs",
    "outputs",
}
TEXT_PATTERNS = (
    (re.compile(r"/home/[A-Za-z0-9_.-]+/"), "absolute /home path"),
    (re.compile(r"/media/[A-Za-z0-9_.-]+/"), "absolute /media path"),
    (
        re.compile(
            r"(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key)"
            r"\s*[:=]\s*['\"][^'\"]+['\"]"
        ),
        "possible credential assignment",
    ),
    (re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"), "private key"),
)


def _candidate_paths(root: Path) -> list[Path]:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = []
    for relative in completed.stdout.splitlines():
        if relative in PUBLIC_ROOT_FILES or relative.startswith(PUBLIC_PREFIXES):
            paths.append(root / relative)
    return paths


def scan_public_tree(root: str | Path) -> list[str]:
    root = Path(root).resolve()
    issues: list[str] = []
    for path in _candidate_paths(root):
        relative = path.relative_to(root)
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(f"{relative}: generated/weight artifact")
            continue
        if FORBIDDEN_PARTS.intersection(relative.parts):
            issues.append(f"{relative}: generated directory")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(f"{relative}: unexpected binary file")
            continue
        for pattern, description in TEXT_PATTERNS:
            if pattern.search(text):
                issues.append(f"{relative}: {description}")
    return sorted(set(issues))


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    issues = scan_public_tree(root)
    if issues:
        print("Public repository hygiene check failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("FutureKV public surfaces are clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
