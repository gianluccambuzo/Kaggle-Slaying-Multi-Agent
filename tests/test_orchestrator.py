from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kagglers.core.budget import Budget, BudgetLimits
from kagglers.core.orchestrator import Orchestrator
from kagglers.core.sandbox import SandboxLimits
from kagglers.core.store import Store
from kagglers.core.types import Candidate
from kagglers.packs import get_pack
from tests.fakes import FakeKaggleClient

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
FAST_LIMITS = SandboxLimits(timeout_s=120, memory_mb=4000, cpu_threads=1)
FAST_PARAMS = {
    "seeds": [0],
    "lgbm": {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 15,
             "min_child_samples": 10, "verbosity": -1},
}


class CountingEvaluator:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def evaluate(self, candidate):
        self.calls += 1
        return self.inner.evaluate(candidate)


def _pack(tmp_path: Path, limits: SandboxLimits = FAST_LIMITS):
    pack = get_pack(
        "airline_s6e10",
        workdir=tmp_path / "work",
        data_dir=tmp_path / "data",
        positive_label="satisfied",
        limits=limits,
    )
    pack.baseline_params = FAST_PARAMS
    pack.evaluator = CountingEvaluator(pack.evaluator)
    return pack


def _orch(pack, tmp_path, kaggle, compute_minutes: float = 60.0):
    store = Store(tmp_path / "kagglers.db")
    budget = Budget(BudgetLimits(llm_usd=1.0, compute_minutes=compute_minutes), pack.rules)
    orch = Orchestrator(pack, store, budget, kaggle, tmp_path / "runs", now=lambda: NOW)
    return orch, store


def test_full_phase1_run_reaches_stop_with_submission_and_snapshot(synthetic_comp, tmp_path):
    pack = _pack(tmp_path)
    kaggle = FakeKaggleClient(fixture_dir=synthetic_comp)
    orch, store = _orch(pack, tmp_path, kaggle)

    run_id = orch.run()

    run = store.get_run(run_id)
    assert run["stop_reason"] == "phase1_complete"
    assert run["state"] == "STOP"
    assert run["finished_at"] is not None
    exps = store.list_experiments(run_id)
    assert len(exps) == 1
    assert exps[0]["status"] == "ok"
    assert exps[0]["operator"] == "baseline"
    assert exps[0]["cv_mean"] > 0.7
    subs = store.list_submissions(run_id)
    assert len(subs) == 1
    assert subs[0]["kaggle_ref"] == "1"
    assert subs[0]["message"] == f"kagglers {exps[0]['exp_id']}"
    assert subs[0]["status"] == "complete"
    assert subs[0]["lb_public"] == pytest.approx(0.95)
    assert subs[0]["cv_at_submit"] == pytest.approx(exps[0]["cv_mean"])
    snaps = store.list_leaderboard_snapshots(run_id)
    assert len(snaps) == 1
    assert snaps[0]["n_teams"] == 10
    assert snaps[0]["our_score"] == pytest.approx(0.95)
    assert snaps[0]["percentil"] == pytest.approx(10.0)
    assert len(kaggle.submitted) == 1
    csv = orch.submission_path(run_id)
    assert csv.exists()
    assert kaggle.submitted[0][1] == csv
    assert (pack.data_dir / "folds_k5_s42.npy").exists()


def test_resume_after_crash_in_monitor_does_not_retrain(synthetic_comp, tmp_path):
    pack = _pack(tmp_path)
    kaggle = FakeKaggleClient(fixture_dir=synthetic_comp, fail_leaderboard_once=True)
    orch, store = _orch(pack, tmp_path, kaggle)

    run_id = orch.run()
    run = store.get_run(run_id)
    assert run["stop_reason"] == "error: KaggleTransientError"
    assert run["state"] == "SUBMIT"  # MONITOR not marked complete
    assert store.list_leaderboard_snapshots(run_id) == []
    store.close()

    pack2 = _pack(tmp_path)
    orch2, store2 = _orch(pack2, tmp_path, kaggle)
    assert orch2.run(run_id) == run_id

    run = store2.get_run(run_id)
    assert run["stop_reason"] == "phase1_complete"
    assert len(store2.list_experiments(run_id)) == 1
    assert pack2.evaluator.calls == 0
    assert len(kaggle.submitted) == 1
    assert len(store2.list_leaderboard_snapshots(run_id)) == 1


def test_resume_after_submit_success_does_not_resubmit(synthetic_comp, tmp_path):
    pack = _pack(tmp_path)
    kaggle = FakeKaggleClient(fixture_dir=synthetic_comp, fail_submit_after_success=True)
    orch, store = _orch(pack, tmp_path, kaggle)

    run_id = orch.run()
    run = store.get_run(run_id)
    assert run["stop_reason"] == "error: KaggleError"
    assert run["state"] == "BASELINE"
    assert store.list_submissions(run_id) == []
    assert len(kaggle.submitted) == 1
    store.close()

    pack2 = _pack(tmp_path)
    orch2, store2 = _orch(pack2, tmp_path, kaggle)
    orch2.run(run_id)

    assert len(kaggle.submitted) == 1
    subs = store2.list_submissions(run_id)
    assert len(subs) == 1
    assert subs[0]["kaggle_ref"] == "1"
    assert store2.get_run(run_id)["stop_reason"] == "phase1_complete"
    assert pack2.evaluator.calls == 0


def test_quota_exhausted_stops_without_submitting(synthetic_comp, tmp_path):
    pack = _pack(tmp_path)
    pack.rules = replace(pack.rules, daily_submission_limit=1)
    kaggle = FakeKaggleClient(fixture_dir=synthetic_comp)
    orch, store = _orch(pack, tmp_path, kaggle)
    # A submission already made today for the same competition, by another run.
    other = store.create_run(pack.competition)
    store.add_experiment(other, Candidate("old", "baseline", {"x": 1}), "earlier")
    store.add_submission("old", "r0", "kagglers old", 0.9, submitted_at=NOW)

    run_id = orch.run()

    run = store.get_run(run_id)
    assert run["stop_reason"] == "quota_exhausted"
    assert run["state"] == "BASELINE"
    assert kaggle.submitted == []
    assert store.list_submissions(run_id) == []
    assert orch.submission_path(run_id).exists()  # CSV built and validated anyway


def test_failed_baseline_stops_with_evaluation_failed(synthetic_comp, tmp_path):
    pack = _pack(tmp_path, SandboxLimits(timeout_s=0.01, memory_mb=4000, cpu_threads=1))
    kaggle = FakeKaggleClient(fixture_dir=synthetic_comp)
    orch, store = _orch(pack, tmp_path, kaggle)

    run_id = orch.run()

    run = store.get_run(run_id)
    assert run["stop_reason"] == "evaluation_failed"
    assert run["state"] == "PREPARE"
    exps = store.list_experiments(run_id)
    assert len(exps) == 1
    assert exps[0]["status"] == "timeout"
    assert exps[0]["cv_mean"] is None
    assert kaggle.submitted == []
    store.close()

    # Resuming does not retry a config that already failed (dedupe by config_hash).
    pack2 = _pack(tmp_path)
    orch2, store2 = _orch(pack2, tmp_path, kaggle)
    orch2.run(run_id)
    assert store2.get_run(run_id)["stop_reason"] == "evaluation_failed"
    assert pack2.evaluator.calls == 0
    assert len(store2.list_experiments(run_id)) == 1


def test_rules_not_accepted_stops_with_clear_reason(synthetic_comp, tmp_path):
    pack = _pack(tmp_path)
    kaggle = FakeKaggleClient(fixture_dir=synthetic_comp, rules_not_accepted=True)
    calls = []
    original = kaggle.download
    kaggle.download = lambda *a, **k: (calls.append(a), original(*a, **k))

    orch, store = _orch(pack, tmp_path, kaggle)
    run_id = orch.run()

    run = store.get_run(run_id)
    assert run["stop_reason"] == "rules_not_accepted"
    assert run["state"] == "INIT"
    assert len(calls) == 1  # no retry loop
    assert store.list_experiments(run_id) == []
    assert pack.evaluator.calls == 0


def test_compute_budget_stops_after_baseline_and_resume_reuses_score(synthetic_comp, tmp_path):
    pack = _pack(tmp_path)
    kaggle = FakeKaggleClient(fixture_dir=synthetic_comp)
    orch, store = _orch(pack, tmp_path, kaggle, compute_minutes=1e-6)

    run_id = orch.run()
    run = store.get_run(run_id)
    assert run["stop_reason"] == "compute_budget"
    assert run["state"] == "BASELINE"
    assert kaggle.submitted == []
    store.close()

    # Same budget on resume: compute already spent by the run counts, so it stops again.
    pack2 = _pack(tmp_path)
    orch2, store2 = _orch(pack2, tmp_path, kaggle, compute_minutes=1e-6)
    orch2.run(run_id)
    assert store2.get_run(run_id)["stop_reason"] == "compute_budget"
    store2.close()

    pack3 = _pack(tmp_path)
    orch3, store3 = _orch(pack3, tmp_path, kaggle, compute_minutes=60)
    orch3.run(run_id)
    assert store3.get_run(run_id)["stop_reason"] == "phase1_complete"
    assert pack2.evaluator.calls == 0
    assert pack3.evaluator.calls == 0
    exp = store3.list_experiments(run_id)[0]
    sub = store3.list_submissions(run_id)[0]
    assert sub["cv_at_submit"] == pytest.approx(exp["cv_mean"])
    assert len(kaggle.submitted) == 1


def test_missing_public_score_records_snapshot_without_percentile(synthetic_comp, tmp_path):
    pack = _pack(tmp_path)
    kaggle = FakeKaggleClient(fixture_dir=synthetic_comp, public_score=None)
    orch, store = _orch(pack, tmp_path, kaggle)

    run_id = orch.run()

    assert store.get_run(run_id)["stop_reason"] == "phase1_complete"
    snaps = store.list_leaderboard_snapshots(run_id)
    assert len(snaps) == 1
    assert snaps[0]["n_teams"] == 10
    assert snaps[0]["our_score"] is None
    assert snaps[0]["percentil"] is None


def test_resume_of_completed_run_does_nothing(synthetic_comp, tmp_path):
    pack = _pack(tmp_path)
    kaggle = FakeKaggleClient(fixture_dir=synthetic_comp)
    orch, store = _orch(pack, tmp_path, kaggle)
    run_id = orch.run()
    calls_before = kaggle.leaderboard_calls

    orch.run(run_id)

    assert kaggle.leaderboard_calls == calls_before
    assert len(kaggle.submitted) == 1
    assert len(store.list_leaderboard_snapshots(run_id)) == 1
    assert store.get_run(run_id)["stop_reason"] == "phase1_complete"


def test_resume_rejects_run_of_another_competition(synthetic_comp, tmp_path):
    pack = _pack(tmp_path)
    orch, store = _orch(pack, tmp_path, FakeKaggleClient(fixture_dir=synthetic_comp))
    other = store.create_run("some-other-comp")
    with pytest.raises(ValueError, match="some-other-comp"):
        orch.run(other)


def test_high_adversarial_auc_logs_warning(synthetic_comp, tmp_path, caplog):
    pack = _pack(tmp_path)
    original = pack.prepare

    def prepare():
        original()
        pack.adversarial_auc = 0.9

    pack.prepare = prepare
    orch, _ = _orch(pack, tmp_path, FakeKaggleClient(fixture_dir=synthetic_comp))
    with caplog.at_level("WARNING", logger="kagglers.orchestrator"):
        orch.run()
    assert any("adversarial" in r.getMessage().lower() for r in caplog.records)


def test_crash_during_baseline_is_resumable_and_reuses_exp_id(synthetic_comp, tmp_path):
    pack = _pack(tmp_path)

    class Crashing:
        def evaluate(self, candidate):
            raise RuntimeError("process died mid-training")

    pack.evaluator = Crashing()
    kaggle = FakeKaggleClient(fixture_dir=synthetic_comp)
    orch, store = _orch(pack, tmp_path, kaggle)

    run_id = orch.run()
    run = store.get_run(run_id)
    assert run["stop_reason"] == "error: RuntimeError"
    assert run["state"] == "PREPARE"
    (exp,) = store.list_experiments(run_id)
    assert exp["status"] == "running"
    store.close()

    pack2 = _pack(tmp_path)
    orch2, store2 = _orch(pack2, tmp_path, kaggle)
    orch2.run(run_id)
    assert store2.get_run(run_id)["stop_reason"] == "phase1_complete"
    assert pack2.evaluator.calls == 1
    (exp2,) = store2.list_experiments(run_id)
    assert exp2["exp_id"] == exp["exp_id"]
    assert exp2["status"] == "ok"
    assert store2.list_submissions(run_id)[0]["message"] == f"kagglers {exp['exp_id']}"
