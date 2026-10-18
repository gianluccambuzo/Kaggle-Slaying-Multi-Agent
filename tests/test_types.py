import re

from kagglers.core.types import Candidate


def test_config_hash_is_stable_across_param_order():
    a = Candidate("a", "baseline", {"x": 1, "y": 2})
    b = Candidate("b", "baseline", {"y": 2, "x": 1})
    assert a.config_hash == b.config_hash


def test_config_hash_differs_on_params():
    a = Candidate("a", "baseline", {"x": 1})
    b = Candidate("a", "baseline", {"x": 2})
    assert a.config_hash != b.config_hash


def test_config_hash_has_16_hex_chars():
    h = Candidate("a", "baseline", {"x": 1}).config_hash
    assert re.fullmatch(r"[0-9a-f]{16}", h)


def test_evaluation_failed_lives_in_core_and_is_reexported_by_airline_evaluator():
    from kagglers.core.types import EvaluationFailed
    from kagglers.packs.airline_s6e10.evaluator import EvaluationFailed as AirlineEF

    assert AirlineEF is EvaluationFailed
    err = EvaluationFailed("timeout", "boom")
    assert (err.status, err.stderr) == ("timeout", "boom")
