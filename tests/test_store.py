from datetime import datetime, timedelta, timezone

from kagglers.core.store import Store
from kagglers.core.types import Candidate


def _cand(exp_id="e1", **params):
    return Candidate(exp_id=exp_id, operator="baseline", params=params or {"lr": 0.1})


def test_create_run_starts_in_init_state(tmp_path):
    store = Store(tmp_path / "s.db")
    run_id = store.create_run("comp")
    run = store.get_run(run_id)
    assert run["state"] == "INIT"
    assert run["competition"] == "comp"


def test_set_state_persists_across_reopen(tmp_path):
    db = tmp_path / "s.db"
    store = Store(db)
    run_id = store.create_run("comp")
    store.set_state(run_id, "BASELINE")
    store.close()
    assert Store(db).get_run(run_id)["state"] == "BASELINE"


def test_add_experiment_rejects_duplicate_config_hash(tmp_path):
    store = Store(tmp_path / "s.db")
    run_id = store.create_run("comp")
    assert store.add_experiment(run_id, _cand("a"), "h") is True
    assert store.add_experiment(run_id, _cand("b"), "h") is False
    assert len(store.list_experiments(run_id)) == 1


def test_finish_experiment_records_cv_and_status(tmp_path):
    store = Store(tmp_path / "s.db")
    run_id = store.create_run("comp")
    store.add_experiment(run_id, _cand("a"), "h")
    store.finish_experiment(
        "a", status="ok", cv_mean=0.9, cv_std=0.01, seeds=3, compute_s=12.5
    )
    exp = store.get_experiment("a")
    assert exp["status"] == "ok"
    assert exp["cv_mean"] == 0.9
    assert exp["cv_std"] == 0.01
    assert exp["seeds"] == 3
    assert exp["promoted"] == 0
    assert store.get_experiment("missing") is None


def test_count_submissions_today_counts_only_current_utc_day(tmp_path):
    store = Store(tmp_path / "s.db")
    run_id = store.create_run("comp")
    store.add_experiment(run_id, _cand("a"), "h")
    now = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    store.add_submission("a", "r1", "m", 0.9, submitted_at=now - timedelta(days=1))
    sub = store.add_submission("a", "r2", "m", 0.9, submitted_at=now)
    assert store.count_submissions_today("comp", now) == 1
    store.update_submission(sub, status="complete", lb_public=0.88)
    subs = store.list_submissions(run_id)
    assert len(subs) == 2
    assert {s["lb_public"] for s in subs} == {None, 0.88}
    store.add_leaderboard_snapshot(run_id, 100, 0.88, 0.5)


def test_list_leaderboard_snapshots_returns_rows_for_run(tmp_path):
    store = Store(tmp_path / "s.db")
    run_id = store.create_run("comp")
    other = store.create_run("comp")
    store.add_leaderboard_snapshot(run_id, 100, 0.88, 12.0)
    store.add_leaderboard_snapshot(run_id, 101, None, None)
    store.add_leaderboard_snapshot(other, 5, 0.5, 80.0)
    snaps = store.list_leaderboard_snapshots(run_id)
    assert [(s["n_teams"], s["our_score"], s["percentil"]) for s in snaps] == [
        (100, 0.88, 12.0),
        (101, None, None),
    ]
