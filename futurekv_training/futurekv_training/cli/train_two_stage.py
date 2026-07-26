"""Build a filtered stage-two dataset and continue judge training."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from futurekv_training.checkpoint import validate_checkpoint
from futurekv_training.cli.train_judge import parse_args as parse_judge_args
from futurekv_training.cli.train_judge import train
from futurekv_training.trainers.two_stage import filter_correct_generations


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_stage_two_dataset(
    source_dataset: str | Path,
    raw_generations: str | Path,
    output_path: str | Path,
) -> tuple[list[dict], dict[str, int]]:
    source_rows = _load_json(Path(source_dataset))
    generated_rows, skipped = filter_correct_generations(
        _load_jsonl(Path(raw_generations))
    )
    source_by_id = {
        str(row.get("id", index)): row
        for index, row in enumerate(source_rows)
    }
    filtered = []
    missing_source = 0
    for generated in generated_rows:
        source_id = str(generated.get("base_id", generated.get("id")))
        source = source_by_id.get(source_id)
        if source is None:
            missing_source += 1
            continue
        merged = dict(source)
        messages = [dict(message) for message in merged["messages"]]
        response = str(generated["response"])
        if len(messages) >= 2:
            messages[1] = {"role": "assistant", "content": response}
        else:
            messages.append({"role": "assistant", "content": response})
        merged["messages"] = messages
        merged["stage2_generation"] = {
            "sample_uid": generated.get("sample_uid"),
            "checkpoint_method": generated.get("kv_method", "futurekv"),
        }
        filtered.append(merged)
    if missing_source:
        skipped = dict(Counter(skipped) + Counter({"missing_source": missing_source}))
    if not filtered:
        raise ValueError("No correct stage-two samples remain after filtering.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return filtered, skipped


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter fresh FutureKV generation JSONL and continue judge training "
            "from a validated stage-one checkpoint."
        )
    )
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--raw-generations", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--filtered-dataset", required=True)
    parser.add_argument("--skip-report", required=True)
    known, judge_argv = parser.parse_known_args(argv)
    judge_args = parse_judge_args(
        ["--dataset", known.filtered_dataset, *judge_argv]
    )
    for key, value in vars(known).items():
        setattr(judge_args, key, value)
    return judge_args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_checkpoint(args.stage1_checkpoint)
    _, skipped = build_stage_two_dataset(
        args.source_dataset,
        args.raw_generations,
        args.filtered_dataset,
    )
    report_path = Path(args.skip_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(skipped, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.dataset = args.filtered_dataset
    args.init_checkpoint = args.stage1_checkpoint
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
