import subprocess
from pathlib import Path

import pytest

from kagglers.core.kaggle_client import (
    CliKaggleClient,
    KaggleError,
    KaggleRulesNotAccepted,
    KaggleTransientError,
    SubmissionInfo,
)


def _result(rc=0, out="", err=""):
    return subprocess.CompletedProcess([], rc, stdout=out, stderr=err)


class Runner:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, cmd):
        self.calls.append(cmd)
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


def test_submit_builds_expected_command(tmp_path):
    r = Runner(_result())
    CliKaggleClient(runner=r).submit("comp", tmp_path / "s.csv", "msg")
    cmd = r.calls[0]
    for tok in ("competitions", "submit", "-f", "-m", "--wait"):
        assert tok in cmd


def test_transient_failure_retries_with_backoff_then_raises(tmp_path):
    r = Runner(_result(1, err="500 server error"))
    sleeps = []
    c = CliKaggleClient(runner=r, max_retries=3, sleep=sleeps.append)
    with pytest.raises(KaggleTransientError):
        c.submit("comp", tmp_path / "s.csv", "m")
    assert len(r.calls) == 3
    assert len(sleeps) >= 2 and sleeps == sorted(sleeps) and sleeps[0] < sleeps[-1]


def test_rules_not_accepted_raises_without_retry(tmp_path):
    r = Runner(_result(1, err="403 Forbidden - you must accept the rules"))
    c = CliKaggleClient(runner=r, sleep=lambda s: None)
    with pytest.raises(KaggleRulesNotAccepted):
        c.download("comp", tmp_path)
    assert len(r.calls) == 1


def test_submissions_parses_verbose_csv():
    out = (
        "ref,fileName,date,description,status,publicScore,privateScore\n"
        "1,a.csv,2026-01-01,run1_c1,complete,0.91,\n"
        "2,b.csv,2026-01-02,run1_c2,pending,,\n"
    )
    subs = CliKaggleClient(runner=Runner(_result(out=out))).submissions("comp")
    assert subs == [
        SubmissionInfo("1", "run1_c1", "complete", 0.91),
        SubmissionInfo("2", "run1_c2", "pending", None),
    ]


def test_submissions_tolerates_extra_and_missing_columns():
    out = "ref,description,status,publicScore,extra\n7,d,complete,0.5,x\n"
    subs = CliKaggleClient(runner=Runner(_result(out=out))).submissions("comp")
    assert subs == [SubmissionInfo("7", "d", "complete", 0.5)]


def test_non_transient_error_raises_kaggle_error(tmp_path):
    r = Runner(_result(1, err="404 not found"))
    with pytest.raises(KaggleError):
        CliKaggleClient(runner=r, sleep=lambda s: None).leaderboard("c", tmp_path)


def test_accept_encoding_in_503_is_transient_not_rules(tmp_path):
    r = Runner(_result(1, err="503 Server Error\nAccept-Encoding: gzip\npath /tmp/x403y"))
    c = CliKaggleClient(runner=r, max_retries=3, sleep=lambda s: None)
    with pytest.raises(KaggleTransientError):
        c.download("comp", tmp_path)
    assert len(r.calls) == 3


def test_403_forbidden_raises_rules_single_call(tmp_path):
    r = Runner(_result(1, err="403 - Forbidden"))
    with pytest.raises(KaggleRulesNotAccepted):
        CliKaggleClient(runner=r, sleep=lambda s: None).download("comp", tmp_path)
    assert len(r.calls) == 1


def test_accept_competition_rules_raises_rules_single_call(tmp_path):
    r = Runner(_result(1, err="You must accept the competition rules"))
    with pytest.raises(KaggleRulesNotAccepted):
        CliKaggleClient(runner=r, sleep=lambda s: None).download("comp", tmp_path)
    assert len(r.calls) == 1
