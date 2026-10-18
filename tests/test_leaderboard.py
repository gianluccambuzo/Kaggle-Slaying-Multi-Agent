import pytest

from kagglers.core.leaderboard import top_percent


def _csv(tmp_path, scores):
    p = tmp_path / "lb.csv"
    lines = ["TeamId,TeamName,SubmissionDate,Score"]
    for i, s in enumerate(scores):
        lines.append(f"{i},team{i},2026-01-01 00:00:00,{s}")
    p.write_text("\n".join(lines) + "\n")
    return p


def test_top_percent_best_score_is_one_over_n(tmp_path):
    lb = _csv(tmp_path, [0.5 + i * 0.01 for i in range(10)])
    assert top_percent(0.99, lb) == 10.0


def test_top_percent_ties_do_not_count_as_better(tmp_path):
    lb = _csv(tmp_path, [0.9, 0.9, 0.8, 0.7])
    assert top_percent(0.9, lb) == 25.0


def test_top_percent_lower_is_better(tmp_path):
    lb = _csv(tmp_path, [0.1, 0.2, 0.3, 0.4])
    assert top_percent(0.15, lb, higher_is_better=False) == 50.0


def test_top_percent_empty_leaderboard_raises(tmp_path):
    lb = _csv(tmp_path, [])
    with pytest.raises(ValueError):
        top_percent(0.9, lb)
