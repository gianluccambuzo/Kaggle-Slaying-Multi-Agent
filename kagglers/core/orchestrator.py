"""Phase-1 orchestrator: INIT -> PREPARE -> BASELINE -> SUBMIT -> MONITOR -> STOP.

`runs.state` holds the LAST COMPLETED state. A state is marked complete only when
it finishes; a failure records the stop reason and leaves `runs.state` untouched,
so `run(run_id)` re-runs the failed state and skips the completed ones.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from kagglers.core.budget import Budget
from kagglers.core.kaggle_client import (
    KaggleClient,
    KaggleError,
    KaggleRulesNotAccepted,
    classify_status,
)
from kagglers.core.leaderboard import top_percent
from kagglers.core.sandbox import SandboxOutputError
from kagglers.core.store import Store
from kagglers.core.types import Candidate, EvaluationFailed, Pack, Score

log = logging.getLogger("kagglers.orchestrator")

STATES = ("INIT", "PREPARE", "BASELINE", "SUBMIT", "MONITOR", "STOP")
BASELINE_OPERATOR = "baseline"
ADVERSARIAL_ALERT = 0.6
SCORE_MANIFEST = "score.json"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _Stop(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Orchestrator:
    def __init__(
        self,
        pack: Pack,
        store: Store,
        budget: Budget,
        kaggle: KaggleClient,
        work_dir: Path,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        self.pack = pack
        self.store = store
        self.budget = budget
        self.kaggle = kaggle
        self.work_dir = Path(work_dir)
        self.now = now
        self._handlers = {
            "PREPARE": self._prepare,
            "BASELINE": self._baseline,
            "SUBMIT": self._submit,
            "MONITOR": self._monitor,
        }

    # ---- paths -------------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        return self.work_dir / run_id

    def submission_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "submission" / "submission.csv"

    def _manifest_path(self, run_id: str, exp_id: str) -> Path:
        return self.run_dir(run_id) / exp_id / SCORE_MANIFEST

    @staticmethod
    def submission_message(exp_id: str) -> str:
        return f"kagglers {exp_id}"

    # ---- main loop ---------------------------------------------------------
    def run(self, run_id: str | None = None) -> str:
        if run_id is None:
            run_id = self.store.create_run(self.pack.competition)  # INIT done
            log.info("run %s created for %s", run_id, self.pack.competition)
        else:
            run = self.store.get_run(run_id)
            if run["competition"] != self.pack.competition:
                raise ValueError(
                    f"run {run_id} belongs to competition {run['competition']!r}, "
                    f"not {self.pack.competition!r}"
                )
            # Compute already spent by this run counts against this budget.
            spent = sum(e["compute_s"] or 0.0 for e in self.store.list_experiments(run_id))
            if spent:
                self.budget.charge_compute(spent)
            log.info("resuming run %s after state %s", run_id, run["state"])
        self.run_dir(run_id).mkdir(parents=True, exist_ok=True)

        done = self.store.get_run(run_id)["state"]
        if done == "STOP":
            return run_id
        current = done
        try:
            for state in STATES[STATES.index(done) + 1 : -1]:
                exhausted = self.budget.exhausted()
                if exhausted:
                    raise _Stop(exhausted)
                current = state
                self._handlers[state](run_id)
                self.store.set_state(run_id, state)
                log.info("state %s complete", state)
        except _Stop as stop:
            reason = stop.reason
        except KaggleRulesNotAccepted as exc:
            log.error("competition rules not accepted (state %s): %s", current, exc)
            reason = "rules_not_accepted"
        except Exception as exc:  # leave the run resumable
            log.exception("unexpected error in state %s", current)
            reason = f"error: {type(exc).__name__}"
        else:
            self.store.finish_run(run_id, "phase1_complete")
            self.store.set_state(run_id, "STOP")
            return run_id
        log.warning("run %s stopped in state %s: %s", run_id, current, reason)
        self.store.finish_run(run_id, reason)
        return run_id

    # ---- states ------------------------------------------------------------
    def _prepare(self, run_id: str) -> None:
        self.kaggle.download(self.pack.competition, self.pack.data_dir)
        self.pack.prepare()
        adv = getattr(self.pack, "adversarial_auc", None)
        if adv is not None and adv > ADVERSARIAL_ALERT:
            log.warning(
                "adversarial validation AUC %.3f > %.2f: train and test distributions differ",
                adv,
                ADVERSARIAL_ALERT,
            )

    def _baseline_row(self, run_id: str) -> dict | None:
        probe = Candidate("", BASELINE_OPERATOR, self.pack.baseline_params)
        for exp in self.store.list_experiments(run_id):
            if exp["config_hash"] == probe.config_hash:
                return exp
        return None

    def _baseline(self, run_id: str) -> None:
        existing = self._baseline_row(run_id)
        if existing is not None and existing["status"] == "ok":
            log.info("baseline %s already evaluated; reusing its score", existing["exp_id"])
            self._load_score(run_id, existing["exp_id"])  # artifacts must still exist
            return
        if existing is not None and existing["status"] in ("failed", "timeout"):
            raise _Stop("evaluation_failed")
        if existing is not None:  # status 'running': interrupted mid-evaluation
            exp_id = existing["exp_id"]
        else:
            exp_id = uuid.uuid4().hex
        candidate = Candidate(exp_id, BASELINE_OPERATOR, self.pack.baseline_params)
        if existing is None:
            self.store.add_experiment(run_id, candidate, "phase 1 baseline")
        seeds = len(candidate.params.get("seeds", [0])) or 1

        t0 = time.monotonic()
        try:
            score = self.pack.evaluator.evaluate(candidate)
        except (EvaluationFailed, SandboxOutputError) as exc:
            elapsed = time.monotonic() - t0
            status = "timeout" if getattr(exc, "status", None) == "timeout" else "failed"
            log.error("baseline evaluation %s: %s", status, exc)
            self.store.finish_experiment(
                exp_id, status=status, cv_mean=None, cv_std=None, seeds=seeds, compute_s=elapsed
            )
            self.budget.charge_compute(elapsed)
            raise _Stop("evaluation_failed") from exc
        elapsed = time.monotonic() - t0

        self._save_score(run_id, exp_id, score)
        self.store.finish_experiment(
            exp_id,
            status="ok",
            cv_mean=score.value,
            cv_std=score.std,
            seeds=seeds,
            compute_s=elapsed,
        )
        self.budget.charge_compute(elapsed)
        log.info("baseline CV %s = %.5f ± %.5f", self.pack.rules.metric_name, score.value, score.std)

    def _submit(self, run_id: str) -> None:
        exp = self._baseline_row(run_id)
        if exp is None or exp["status"] != "ok":
            raise RuntimeError(f"no successful baseline experiment in run {run_id}")
        exp_id = exp["exp_id"]
        score = self._load_score(run_id, exp_id)
        artifact = self.pack.packager.build(score, self.submission_path(run_id).parent)
        message = self.submission_message(exp_id)
        comp = self.pack.competition

        if any(s["message"] == message for s in self.store.list_submissions(run_id)):
            self._stop_if_rejected(self._refresh_submission(run_id, message))
            return
        remote = self._find_remote(message)
        if remote is not None:  # submitted before a crash, never recorded (Review Focus 5)
            log.warning("submission %r already on Kaggle (ref %s); reconciling", message, remote.ref)
            self._record_submission(exp_id, remote, message, score.value)
            self._stop_if_rejected(self._submission_row(run_id, message))
            return

        today = self.store.count_submissions_today(comp, self.now())
        if not self.budget.can_submit(today, final=True):
            raise _Stop("quota_exhausted")
        self.kaggle.submit(comp, artifact.path, message)
        remote = self._find_remote(message)
        if remote is None:
            raise KaggleError(f"submission {message!r} accepted but not listed by Kaggle")
        self._record_submission(exp_id, remote, message, score.value)
        self._stop_if_rejected(self._submission_row(run_id, message))

    def _monitor(self, run_id: str) -> None:
        subs = self.store.list_submissions(run_id)
        if not subs:
            raise RuntimeError(f"no submission recorded for run {run_id}")
        sub = subs[-1]
        if sub["lb_public"] is None:
            sub = self._refresh_submission(run_id, sub["message"]) or sub
        self._stop_if_rejected(sub)
        lb_csv = self.kaggle.leaderboard(self.pack.competition, self.run_dir(run_id) / "leaderboard")
        n_teams = len(pd.read_csv(lb_csv))
        our = sub["lb_public"]
        if our is None:
            log.warning("public score not available for %s; snapshot without percentile",
                        sub["kaggle_ref"])
            percentile = None
        else:
            percentile = top_percent(our, lb_csv, self.pack.rules.higher_is_better)
            log.info("public LB %.5f -> top %.1f%% of %d teams (metadata only)",
                     our, percentile, n_teams)
        self.store.add_leaderboard_snapshot(run_id, n_teams, our, percentile)

    # ---- helpers -----------------------------------------------------------
    def _find_remote(self, message: str):
        for info in self.kaggle.submissions(self.pack.competition):
            if info.description == message:
                return info
        return None

    def _record_submission(self, exp_id: str, info, message: str, cv: float) -> None:
        sub_id = self.store.add_submission(exp_id, info.ref, message, cv, submitted_at=self.now())
        self.store.update_submission(sub_id, status=info.status, lb_public=info.public_score)

    def _submission_row(self, run_id: str, message: str) -> dict:
        return next(s for s in self.store.list_submissions(run_id) if s["message"] == message)

    def _refresh_submission(self, run_id: str, message: str) -> dict:
        row = self._submission_row(run_id, message)
        info = self._find_remote(message)
        if info is None:
            return row
        self.store.update_submission(row["sub_id"], status=info.status, lb_public=info.public_score)
        return self._submission_row(run_id, message)

    @staticmethod
    def _stop_if_rejected(sub: dict) -> None:
        """A submission Kaggle scored as error/failed ends the run (never resent)."""
        if classify_status(sub["status"] or "") == "error":
            log.error(
                "submission %r (ref %s) rejected by Kaggle: status %s",
                sub["message"], sub["kaggle_ref"], sub["status"],
            )
            raise _Stop("submission_rejected")

    def _save_score(self, run_id: str, exp_id: str, score: Score) -> None:
        path = self._manifest_path(run_id, exp_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "value": score.value,
                    "std": score.std,
                    "n_runs": score.n_runs,
                    "fold_scores": list(score.fold_scores),
                    "oof_path": str(Path(score.oof_path).resolve()),
                    "test_path": str(Path(score.test_path).resolve()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_score(self, run_id: str, exp_id: str) -> Score:
        path = self._manifest_path(run_id, exp_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        score = Score(
            value=data["value"],
            std=data["std"],
            n_runs=data["n_runs"],
            fold_scores=data["fold_scores"],
            oof_path=Path(data["oof_path"]),
            test_path=Path(data["test_path"]),
        )
        for p in (score.oof_path, score.test_path):
            if not p.exists():
                raise FileNotFoundError(f"artifact of experiment {exp_id} missing: {p}")
        return score
