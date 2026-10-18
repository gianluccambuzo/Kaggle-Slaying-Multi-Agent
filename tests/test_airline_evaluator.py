from pathlib import Path

import numpy as np
import pytest

from kagglers.core.sandbox import SandboxLimits
from kagglers.core.types import Candidate
from kagglers.packs.airline_s6e10.config import AirlineConfig
from kagglers.packs.airline_s6e10.data import (
    encode_target,
    load_train_test,
    make_folds,
    read_schema,
    save_folds,
)
from kagglers.packs.airline_s6e10.evaluator import AirlineEvaluator, EvaluationFailed

PARAMS = {
    "seeds": [0],
    "lgbm": {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 15,
             "min_child_samples": 10, "verbosity": -1},
}
LIMITS = SandboxLimits(timeout_s=120, memory_mb=4000, cpu_threads=1)


def _setup(data_dir: Path, tmp_path: Path, limits: SandboxLimits = LIMITS):
    cfg = AirlineConfig(data_dir=data_dir, positive_label="satisfied", n_splits=5, fold_seed=42)
    schema = read_schema(data_dir)
    train, test = load_train_test(data_dir, schema)
    y = encode_target(train[schema.target_col], cfg.positive_label)
    save_folds(make_folds(y, cfg.n_splits, cfg.fold_seed), data_dir / "folds_k5_s42.npy")
    return AirlineEvaluator(cfg, tmp_path / "runs", limits), len(train), len(test)


def test_baseline_evaluate_returns_score_with_oof_and_test_shapes(synthetic_comp, tmp_path):
    ev, n_train, n_test = _setup(synthetic_comp, tmp_path)
    score = ev.evaluate(Candidate("e1", "baseline", PARAMS))
    assert np.load(score.oof_path).shape == (n_train,)
    assert np.load(score.test_path).shape == (n_test,)
    assert score.n_runs == 5
    assert len(score.fold_scores) == 5


def test_baseline_cv_auc_beats_chance_on_synthetic_signal(synthetic_comp, tmp_path):
    ev, _, _ = _setup(synthetic_comp, tmp_path)
    score = ev.evaluate(Candidate("e1", "baseline", PARAMS))
    assert score.value > 0.7


def test_baseline_evaluates_with_unseen_category_and_nulls_in_test(synthetic_comp, tmp_path):
    ev, _, _ = _setup(synthetic_comp, tmp_path)
    score = ev.evaluate(Candidate("e1", "baseline", PARAMS))
    assert not np.isnan(np.load(score.test_path)).any()


def test_evaluator_raises_evaluation_failed_on_sandbox_timeout(synthetic_comp, tmp_path):
    ev, _, _ = _setup(synthetic_comp, tmp_path, SandboxLimits(0.1, 4000, 1))
    with pytest.raises(EvaluationFailed) as ei:
        ev.evaluate(Candidate("e1", "baseline", PARAMS))
    assert ei.value.status == "timeout"


def test_evaluator_raises_evaluation_failed_on_operator_error(synthetic_comp, tmp_path):
    ev, _, _ = _setup(synthetic_comp, tmp_path)
    bad = {"seeds": [0], "lgbm": {"no_such_param_xyz": 1, "boosting_type": "nope"}}
    with pytest.raises(EvaluationFailed) as ei:
        ev.evaluate(Candidate("e1", "baseline", bad))
    assert ei.value.status == "failed"


def test_evaluator_is_reproducible_for_same_seed(synthetic_comp, tmp_path):
    ev, _, _ = _setup(synthetic_comp, tmp_path)
    a = ev.evaluate(Candidate("e1", "baseline", PARAMS))
    b = ev.evaluate(Candidate("e2", "baseline", PARAMS))
    assert a.value == b.value
