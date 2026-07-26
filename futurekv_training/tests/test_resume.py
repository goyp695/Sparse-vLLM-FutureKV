import json
from pathlib import Path

from futurekv_training.trainers.two_stage import completed_sample_ids


def test_resume_counts_only_successful_unique_samples(tmp_path: Path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"id": "a", "status": "success"}),
            json.dumps({"id": "a", "status": "success"}),
            json.dumps({"id": "b", "status": "error"}),
        ])
        + "\n"
    )
    assert completed_sample_ids(path) == {"a"}
