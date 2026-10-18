"""Command line: `kagglers run <pack> --phase1 [--resume RUN_ID] [--compute-minutes N]`."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from kagglers.core.budget import Budget, BudgetLimits
from kagglers.core.kaggle_client import CliKaggleClient, KaggleClient
from kagglers.core.orchestrator import Orchestrator
from kagglers.core.store import Store
from kagglers.packs import UnknownPackError, get_pack

DEFAULT_COMPUTE_MINUTES = 120.0
# Phase 1 never calls an LLM; the cap only has to be > 0 so Budget.exhausted() stays quiet.
PHASE1_LLM_USD = 1.0
# Stop reasons that a resume of the same run cannot change.
FINAL_STOP_REASONS = ("evaluation_failed", "submission_rejected")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kagglers")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a competition pack")
    run.add_argument("pack", help="pack name, e.g. airline_s6e10")
    run.add_argument("--phase1", action="store_true",
                     help="PREPARE -> BASELINE -> SUBMIT -> MONITOR (required)")
    run.add_argument("--resume", metavar="RUN_ID", help="resume an interrupted run")
    run.add_argument("--compute-minutes", type=float, default=DEFAULT_COMPUTE_MINUTES,
                     help=f"compute budget in minutes (default {DEFAULT_COMPUTE_MINUTES:g})")
    run.add_argument("--positive-label",
                     help="target value counted as the positive class (string targets)")
    return p


def _err(msg: str) -> None:
    print(f"kagglers: {msg}", file=sys.stderr)


def main(
    argv: list[str] | None = None,
    *,
    kaggle: KaggleClient | None = None,
    base_dir: Path | None = None,
    pack_kwargs: dict | None = None,
) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:  # argparse already printed the message
        return int(exc.code or 0)
    if not args.phase1:
        _err("only --phase1 is implemented; run with --phase1")
        return 2

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    runs_dir = base / "runs"

    kwargs: dict = {"workdir": base}
    if args.positive_label is not None:
        kwargs["positive_label"] = args.positive_label
    kwargs.update(pack_kwargs or {})
    try:
        pack = get_pack(args.pack, **kwargs)
    except UnknownPackError as exc:
        _err(str(exc))
        return 2

    runs_dir.mkdir(parents=True, exist_ok=True)
    store = Store(runs_dir / "kagglers.db")
    try:
        if args.resume is not None:
            try:
                store.get_run(args.resume)
            except KeyError:
                _err(f"no run with id {args.resume!r} in {runs_dir / 'kagglers.db'}")
                return 2
        budget = Budget(BudgetLimits(llm_usd=PHASE1_LLM_USD,
                                     compute_minutes=args.compute_minutes), pack.rules)
        orch = Orchestrator(pack, store, budget, kaggle or CliKaggleClient(), runs_dir)
        try:
            run_id = orch.run(args.resume)
        except ValueError as exc:
            _err(str(exc))
            return 2
        return _report(orch, store, run_id)
    finally:
        store.close()


def _report(orch: Orchestrator, store: Store, run_id: str) -> int:
    run = store.get_run(run_id)
    print(f"run_id: {run_id}")
    ok = [e for e in store.list_experiments(run_id) if e["status"] == "ok"]
    if ok:
        e = ok[-1]
        print(f"CV {orch.pack.rules.metric_name}: {e['cv_mean']:.5f} ± {e['cv_std']:.5f}")
    csv = orch.submission_path(run_id)
    if csv.exists():
        print(f"submission CSV: {csv}")
    subs = store.list_submissions(run_id)
    if subs:
        s = subs[-1]
        lb = "n/a" if s["lb_public"] is None else f"{s['lb_public']:.5f}"
        print(f"kaggle_ref: {s['kaggle_ref']} (status {s['status']}, public LB {lb})")
    snaps = store.list_leaderboard_snapshots(run_id)
    if snaps:
        pct = snaps[-1]["percentil"]
        print("percentile: n/a" if pct is None else f"percentile: top {pct:.1f}%")
    reason = run["stop_reason"]
    print(f"stop_reason: {reason}")
    if reason == "phase1_complete":
        return 0
    if reason == "submission_rejected" and subs:
        s = subs[-1]
        print(f"submission rejected by Kaggle (status {s['status']}, "
              f"description {s['message']!r}, ref {s['kaggle_ref']})")
    if reason in FINAL_STOP_REASONS:
        print(f"resuming will not help; a new run is needed: kagglers run {orch.pack.name} --phase1")
    else:
        print(f"resume with: kagglers run {orch.pack.name} --phase1 --resume {run_id}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
