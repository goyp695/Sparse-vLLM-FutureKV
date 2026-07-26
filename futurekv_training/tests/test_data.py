import json
from pathlib import Path

import pytest

from futurekv_training.data import load_multimodal_records


def test_data_resolves_images_and_requires_assistant_response(tmp_path: Path):
    image = tmp_path / "figure.png"
    image.touch()
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([{
        "id": "sample-1",
        "messages": [
            {"role": "user", "content": "<image> solve this"},
            {"role": "assistant", "content": "answer"},
        ],
        "images": ["figure.png"],
    }]))

    records = load_multimodal_records(dataset, image_root=tmp_path)

    assert records[0].sample_id == "sample-1"
    assert records[0].image_paths == (image.resolve(),)


@pytest.mark.parametrize("payload", [[], [{}], [{"messages": []}]])
def test_data_rejects_empty_or_invalid_records(tmp_path: Path, payload):
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_multimodal_records(dataset, image_root=tmp_path)


def test_data_rejects_missing_image(tmp_path: Path):
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([{
        "messages": [
            {"role": "user", "content": "<image> question"},
            {"role": "assistant", "content": "answer"},
        ],
        "images": ["missing.png"],
    }]))
    with pytest.raises(FileNotFoundError, match="missing.png"):
        load_multimodal_records(dataset, image_root=tmp_path)
