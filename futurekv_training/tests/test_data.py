import json
from contextlib import ExitStack
from pathlib import Path

import pytest
from PIL import Image

from futurekv_training.cli.train_judge import _processor_messages
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


def test_processor_messages_consumes_numbered_image_markers(tmp_path: Path):
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (2, 2)).save(image_path)
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([{
        "messages": [
            {"role": "user", "content": "<image> first <image1> second"},
            {"role": "assistant", "content": "answer"},
        ],
        "images": ["figure.png", "figure.png"],
    }]))
    record = load_multimodal_records(dataset, image_root=tmp_path)[0]

    with ExitStack() as stack:
        messages = _processor_messages(record, stack)

    image_parts = [
        part
        for message in messages
        for part in message["content"]
        if part["type"] == "image"
    ]
    assert len(image_parts) == 2


def test_processor_messages_preserves_surplus_image_markers(tmp_path: Path):
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (2, 2)).save(image_path)
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([{
        "messages": [
            {"role": "user", "content": "<image> question <image1>"},
            {"role": "assistant", "content": "answer"},
        ],
        "images": ["figure.png"],
    }]))
    record = load_multimodal_records(dataset, image_root=tmp_path)[0]

    with ExitStack() as stack:
        messages = _processor_messages(record, stack)

    parts = messages[0]["content"]
    assert [part["type"] for part in parts].count("image") == 1
    assert {"type": "text", "text": "<image1>"} in parts


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
