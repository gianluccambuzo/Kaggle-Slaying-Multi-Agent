from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from kagglers.core.types import Candidate

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    competition TEXT NOT NULL,
    config_hash TEXT,
    state TEXT NOT NULL DEFAULT 'INIT',
    started_at TEXT NOT NULL,
    finished_at TEXT,
    stop_reason TEXT
);
CREATE TABLE IF NOT EXISTS experiments (
    exp_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    parent_id TEXT REFERENCES experiments(exp_id),
    hypothesis TEXT,
    operator TEXT NOT NULL,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    status TEXT,
    cv_mean REAL,
    cv_std REAL,
    seeds INTEGER,
    delta_vs_best REAL,
    promoted INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    compute_s REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_experiments_run_hash
    ON experiments(run_id, config_hash);
CREATE TABLE IF NOT EXISTS submissions (
    sub_id TEXT PRIMARY KEY,
    exp_id TEXT NOT NULL REFERENCES experiments(exp_id),
    kaggle_ref TEXT,
    message TEXT,
    cv_at_submit REAL,
    lb_public REAL,
    status TEXT,
    submitted_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    ts TEXT NOT NULL,
    n_teams INTEGER,
    our_score REAL,
    percentil REAL
);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _dict_factory(cursor, row):
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


class Store:
    def __init__(self, db_path: Path):
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = _dict_factory
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # runs
    def create_run(self, competition: str) -> str:
        run_id = _new_id()
        with self._conn:
            self._conn.execute(
                "INSERT INTO runs (run_id, competition, state, started_at) "
                "VALUES (?, ?, 'INIT', ?)",
                (run_id, competition, _now().isoformat()),
            )
        return run_id

    def get_run(self, run_id: str) -> dict:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return row

    def set_state(self, run_id: str, state: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET state = ? WHERE run_id = ?", (state, run_id)
            )

    def finish_run(self, run_id: str, stop_reason: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET finished_at = ?, stop_reason = ? WHERE run_id = ?",
                (_now().isoformat(), stop_reason, run_id),
            )

    # experiments
    def add_experiment(
        self,
        run_id: str,
        candidate: Candidate,
        hypothesis: str,
        parent_id: str | None = None,
    ) -> bool:
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO experiments (exp_id, run_id, parent_id, hypothesis, "
                    "operator, config_json, config_hash, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'running')",
                    (
                        candidate.exp_id,
                        run_id,
                        parent_id,
                        hypothesis,
                        candidate.operator,
                        json.dumps(candidate.params, sort_keys=True, default=str),
                        candidate.config_hash,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "config_hash" in str(exc):
                return False
            raise
        return True

    def finish_experiment(
        self,
        exp_id: str,
        *,
        status: str,
        cv_mean: float | None,
        cv_std: float | None,
        seeds: int,
        compute_s: float,
        cost_usd: float = 0.0,
        promoted: bool = False,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE experiments SET status = ?, cv_mean = ?, cv_std = ?, "
                "seeds = ?, compute_s = ?, cost_usd = ?, promoted = ? "
                "WHERE exp_id = ?",
                (
                    status,
                    cv_mean,
                    cv_std,
                    seeds,
                    compute_s,
                    cost_usd,
                    int(promoted),
                    exp_id,
                ),
            )

    def get_experiment(self, exp_id: str) -> dict | None:
        return self._conn.execute(
            "SELECT * FROM experiments WHERE exp_id = ?", (exp_id,)
        ).fetchone()

    def list_experiments(self, run_id: str) -> list[dict]:
        return self._conn.execute(
            "SELECT * FROM experiments WHERE run_id = ? ORDER BY rowid", (run_id,)
        ).fetchall()

    # submissions
    def add_submission(
        self,
        exp_id: str,
        kaggle_ref: str,
        message: str,
        cv_at_submit: float,
        submitted_at: datetime | None = None,
    ) -> str:
        sub_id = _new_id()
        ts = _utc(submitted_at) if submitted_at is not None else _now()
        with self._conn:
            self._conn.execute(
                "INSERT INTO submissions (sub_id, exp_id, kaggle_ref, message, "
                "cv_at_submit, status, submitted_at) VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                (sub_id, exp_id, kaggle_ref, message, cv_at_submit, ts.isoformat()),
            )
        return sub_id

    def update_submission(
        self, sub_id: str, *, status: str, lb_public: float | None
    ) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE submissions SET status = ?, lb_public = ? WHERE sub_id = ?",
                (status, lb_public, sub_id),
            )

    def list_submissions(self, run_id: str) -> list[dict]:
        return self._conn.execute(
            "SELECT s.* FROM submissions s JOIN experiments e ON e.exp_id = s.exp_id "
            "WHERE e.run_id = ? ORDER BY s.rowid",
            (run_id,),
        ).fetchall()

    def count_submissions_today(self, competition: str, now: datetime) -> int:
        day = _utc(now).date()
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        end = datetime.fromordinal(day.toordinal() + 1).replace(tzinfo=timezone.utc)
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM submissions s "
            "JOIN experiments e ON e.exp_id = s.exp_id "
            "JOIN runs r ON r.run_id = e.run_id "
            "WHERE r.competition = ? AND s.submitted_at >= ? AND s.submitted_at < ?",
            (competition, start.isoformat(), end.isoformat()),
        ).fetchone()
        return row["n"]

    # leaderboard
    def add_leaderboard_snapshot(
        self,
        run_id: str,
        n_teams: int,
        our_score: float | None,
        percentile: float | None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO leaderboard_snapshots "
                "(run_id, ts, n_teams, our_score, percentil) VALUES (?, ?, ?, ?, ?)",
                (run_id, _now().isoformat(), n_teams, our_score, percentile),
            )
