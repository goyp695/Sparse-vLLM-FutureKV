from pathlib import Path


def test_training_package_does_not_import_runtime():
    package_root = Path(__file__).parents[1] / "futurekv_training"
    offenders = []
    for path in package_root.rglob("*.py"):
        if "sparsevllm" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == []
