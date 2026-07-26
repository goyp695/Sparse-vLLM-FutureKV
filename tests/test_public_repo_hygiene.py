from pathlib import Path

from scripts.validation.check_public_repo import scan_public_tree


def test_futurekv_public_surfaces_are_clean():
    root = Path(__file__).resolve().parents[1]
    assert scan_public_tree(root) == []
