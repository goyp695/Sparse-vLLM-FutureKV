import pytest

from futurekv_training.trainers.two_stage import (
    choose_cut_position,
    filter_correct_generations,
    has_repeated_tail,
)


def test_filters_correct_rows_and_rejects_duplicate_ids():
    rows = [
        {"id": "a", "response": "answer: \\boxed{2}", "answer": "2"},
        {"id": "b", "response": "answer: \\boxed{4}", "answer": "3"},
    ]
    accepted, skipped = filter_correct_generations(rows)
    assert [row["id"] for row in accepted] == ["a"]
    assert skipped == {"incorrect": 1}

    with pytest.raises(ValueError, match="Duplicate"):
        filter_correct_generations([rows[0], rows[0]])


def test_repeat_tail_and_cut_bounds():
    assert has_repeated_tail("abc " * 20)
    assert not has_repeated_tail("a short, normal response")
    assert choose_cut_position(100, min_tokens=20, future_tokens=16, seed=7) in range(20, 85)
    with pytest.raises(ValueError):
        choose_cut_position(20, min_tokens=10, future_tokens=16, seed=7)
