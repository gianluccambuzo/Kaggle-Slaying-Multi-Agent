# Kaggle-Slaying-Multi-Agent (`kagglers`)

Automated pipeline for Kaggle tabular competitions. Phase 1 runs a fixed cycle for
one competition pack (`airline_s6e10`, Playground Series S6E10):

```
INIT -> PREPARE -> BASELINE -> SUBMIT -> MONITOR -> STOP
```

- **PREPARE** downloads the data (`kaggle competitions download <c> -p <dir>`
  writes `<c>.zip`, which is extracted safely into the data directory), fixes 5
  stratified folds and runs a quick adversarial validation (warning if AUC > 0.6).
- **BASELINE** trains LightGBM on the 5 folds in a sandboxed subprocess and records
  the CV ROC AUC.
- **SUBMIT** builds and validates `submission.csv` (predictions matched to the
  `sample_submission.csv` ids), then submits it once (one per run, only if the daily
  quota allows) with `kaggle competitions submit <c> -f <csv> -m <message>`. The
  upload is never retried blindly: after a transient error the client first looks
  for the message in `kaggle competitions submissions`. It then polls
  `submissions` until the status is final (default 60 polls x 10 s); a submission
  still pending after that is kept as `pending` and MONITOR records no percentile.
- **MONITOR** downloads the leaderboard and records the public-LB percentile. The
  public LB is metadata only; it never drives a decision.

Every state is saved in SQLite (`runs/kagglers.db`) when it completes, so an
interrupted run can be resumed without retraining or resubmitting.

## Install

Python 3.12+. Kaggle CLI `kaggle>=2.2,<3` (installed as a dependency; 2.x has no
`--unzip` / `--wait` flags, so the client unzips and polls itself). Each CLI call
has a timeout (default 30 min).

```
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -e .[dev]
pytest                         # offline test suite, no network
```

## Kaggle setup

1. Create an API token on kaggle.com (Settings -> API -> *Create New Token*) and
   save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
   (Windows: `C:\Users\<you>\.kaggle\kaggle.json`). On Linux/macOS run
   `chmod 600 ~/.kaggle/kaggle.json`.
2. Open the competition page
   (https://www.kaggle.com/competitions/playground-series-s6e10), click
   **Join Competition** and accept the rules. Without this the CLI returns 403 and
   the run stops with `stop_reason: rules_not_accepted` (no retry).
3. Check the CLI works: `kaggle competitions list`.

The token is only used by the orchestrator process; operator code runs in a
subprocess with a minimal environment and never sees it.

## Run

```
kagglers run airline_s6e10 --phase1 [--resume RUN_ID] [--compute-minutes N] [--positive-label LABEL]
```

- Data goes to `data/playground-series-s6e10/`, runs and the database to `runs/`
  (both under the current directory).
- `--positive-label` is needed when the target is a string (e.g. `satisfied`); a
  0/1 target needs nothing. If it is missing, PREPARE fails and the error message
  lists the target's unique values.
- `--compute-minutes` caps the compute budget (default 120).
- The command prints the run id, CV (mean ± std), the CSV path, the `kaggle_ref`,
  the percentile and the stop reason. Exit code 0 only for `phase1_complete`.

Stop reasons: `phase1_complete`, `quota_exhausted`, `evaluation_failed`,
`submission_rejected`, `compute_budget`, `rules_not_accepted`,
`error: <ExceptionName>`. `evaluation_failed` and `submission_rejected` (Kaggle
scored the file as an error; its status and description are printed) are final for
that run: the CLI says a new run is needed. For the others fix the cause and resume
with the printed command:
`kagglers run airline_s6e10 --phase1 --resume <run_id>`. Completed states are
skipped; a submission already on Kaggle (matched by its message
`kagglers <exp_id>`) is reconciled instead of being sent again.

## Manual verification (real Kaggle; not automated)

Requires the token and accepted rules (see above). Not run by the test suite.

1. `kagglers run airline_s6e10 --phase1` (add `--positive-label ...` if the target
   is a string).
2. Expected:
   - files in `data/playground-series-s6e10/` (`train.csv`, `test.csv`,
     `sample_submission.csv`, `folds_k5_s42.npy`);
   - `CV roc_auc: 0.95xxx ± ...` printed (expected range 0.95+);
   - `submission CSV: runs/<run_id>/submission/submission.csv` (validated against
     `sample_submission.csv`);
   - `kaggle_ref: ...` printed and visible under *My Submissions* with message
     `kagglers <exp_id>`;
   - `percentile: top X%` printed and stored in `leaderboard_snapshots`;
   - `stop_reason: phase1_complete`, exit code 0.
3. Resume check: interrupt a fresh run (Ctrl+C) during BASELINE, then rerun with
   `--resume <run_id>`; it must continue from the saved state, and a resumed run
   must never create a second submission.
4. Record the results below.

| Date | CV AUC (mean ± std) | Public LB | Percentile | kaggle_ref |
|------|---------------------|-----------|------------|------------|
| 2026-10-19 | 0.95868 ± 0.00064 | 0.95832 | top 65.8% (2299 teams) | 57381215 |
