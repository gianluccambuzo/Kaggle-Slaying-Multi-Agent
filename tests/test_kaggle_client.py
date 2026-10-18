import subprocess
import zipfile
from pathlib import Path

import pytest

from kagglers.core.kaggle_client import (
    CliKaggleClient,
    KaggleError,
    KaggleRulesNotAccepted,
    KaggleTransientError,
    SubmissionInfo,
    classify_status,
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


SUBS_HEADER = "ref,fileName,date,description,status,publicScore,privateScore\n"
TRAIN_CSV = "id,y\n1,0\n"


def _subs(*rows):
    return SUBS_HEADER + "".join(
        f"{ref},s.csv,2026-01-01,{desc},{status},{score},\n" for ref, desc, status, score in rows
    )


class ScriptedRunner:
    """Answers by subcommand (`download`, `submit`, `submissions`, ...). Each entry is a
    CompletedProcess, an exception instance (raised) or a callable(cmd) -> result;
    the last entry of each list repeats."""

    def __init__(self, **script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def __call__(self, cmd):
        self.calls.append(cmd)
        queue = self.script[cmd[2]]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return item(cmd) if callable(item) else item

    def count(self, sub):
        return sum(1 for c in self.calls if c[2] == sub)


def test_submit_builds_expected_command_without_wait(tmp_path):
    r = ScriptedRunner(submit=[_result()], submissions=[_result(out=_subs(("1", "msg", "COMPLETE", 0.9)))])
    CliKaggleClient(runner=r, sleep=lambda s: None).submit("comp", tmp_path / "s.csv", "msg")
    assert r.calls[0] == ["kaggle", "competitions", "submit", "comp",
                          "-f", str(tmp_path / "s.csv"), "-m", "msg"]
    assert "--wait" not in r.calls[0]


def test_submit_polls_submissions_until_terminal_status(tmp_path):
    r = ScriptedRunner(
        submit=[_result()],
        submissions=[
            _result(out=_subs(("1", "msg", "SubmissionStatus.PENDING", ""))),
            _result(out=_subs(("1", "msg", "PENDING", ""))),
            _result(out=_subs(("1", "msg", "SubmissionStatus.COMPLETE", 0.9))),
            _result(out=_subs(("1", "msg", "SubmissionStatus.COMPLETE", 0.9))),
        ],
    )
    sleeps = []
    c = CliKaggleClient(runner=r, sleep=sleeps.append, max_polls=10, poll_interval_s=7.0)
    c.submit("comp", tmp_path / "s.csv", "msg")
    assert r.count("submit") == 1
    assert r.count("submissions") == 3
    assert sleeps == [7.0, 7.0]


def test_submit_stops_polling_on_error_status(tmp_path):
    r = ScriptedRunner(submit=[_result()],
                       submissions=[_result(out=_subs(("1", "msg", "ERROR", "")))])
    CliKaggleClient(runner=r, sleep=lambda s: None).submit("comp", tmp_path / "s.csv", "msg")
    assert r.count("submissions") == 1


def test_submit_gives_up_polling_after_max_polls(tmp_path):
    r = ScriptedRunner(submit=[_result()],
                       submissions=[_result(out=_subs(("1", "msg", "PENDING", "")))])
    sleeps = []
    c = CliKaggleClient(runner=r, sleep=sleeps.append, max_polls=4, poll_interval_s=1.0)
    c.submit("comp", tmp_path / "s.csv", "msg")  # returns; caller sees 'pending'
    assert r.count("submissions") == 4
    assert r.count("submit") == 1


def test_default_poll_settings_are_bounded():
    c = CliKaggleClient()
    assert c.max_polls == 60
    assert c.poll_interval_s == 10.0


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("SubmissionStatus.COMPLETE", "complete"),
        ("complete", "complete"),
        ("COMPLETE", "complete"),
        ("ERROR", "error"),
        ("SubmissionStatus.ERROR", "error"),
        ("failed", "error"),
        ("PENDING", "pending"),
        ("SubmissionStatus.PENDING", "pending"),
        ("", "pending"),
    ],
)
def test_classify_status(raw, expected):
    assert classify_status(raw) == expected


def _write_zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def test_download_has_no_unzip_flag_and_extracts_competition_zip(tmp_path):
    dest = tmp_path / "data"

    def cli_download(cmd):
        out = Path(cmd[cmd.index("-p") + 1])
        out.mkdir(parents=True, exist_ok=True)
        _write_zip(out / "comp.zip", {"train.csv": TRAIN_CSV, "test.csv": "id\n2\n"})
        return _result(out="Downloading comp.zip")

    r = ScriptedRunner(download=[cli_download])
    CliKaggleClient(runner=r, sleep=lambda s: None).download("comp", dest)
    assert r.calls == [["kaggle", "competitions", "download", "comp", "-p", str(dest)]]
    assert (dest / "train.csv").read_text() == TRAIN_CSV
    assert (dest / "test.csv").exists()


def test_download_without_zip_raises(tmp_path):
    r = ScriptedRunner(download=[_result(out="nothing")])
    with pytest.raises(KaggleError, match="comp.zip"):
        CliKaggleClient(runner=r, sleep=lambda s: None).download("comp", tmp_path)


def test_download_refuses_zip_member_outside_dest(tmp_path):
    dest = tmp_path / "data"
    dest.mkdir()
    _write_zip(dest / "comp.zip", {"../evil.csv": "x"})
    r = ScriptedRunner(download=[_result()])
    with pytest.raises(KaggleError, match="unsafe"):
        CliKaggleClient(runner=r, sleep=lambda s: None).download("comp", dest)
    assert not (tmp_path / "evil.csv").exists()


def test_transient_failure_retries_with_backoff_then_raises(tmp_path):
    r = Runner(_result(1, err="500 server error"))
    sleeps = []
    c = CliKaggleClient(runner=r, max_retries=3, sleep=sleeps.append)
    with pytest.raises(KaggleTransientError):
        c.download("comp", tmp_path)
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
