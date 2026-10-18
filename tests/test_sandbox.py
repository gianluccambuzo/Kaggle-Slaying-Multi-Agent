import json
import time
from pathlib import Path

import numpy as np
import psutil
import pytest

from kagglers.core.sandbox import (
    SandboxLimits,
    SandboxOutputError,
    run_in_sandbox,
    validate_operator_output,
)

SCRIPTS = Path(__file__).parent / "sandbox_scripts"
LIMITS = SandboxLimits(timeout_s=20, memory_mb=1000, cpu_threads=1)


def test_ok_script_returns_ok_and_stdout(tmp_path):
    res = run_in_sandbox([str(SCRIPTS / "ok.py")], tmp_path, LIMITS)
    assert res.status == "ok"
    assert res.returncode == 0
    assert "hello sandbox" in res.stdout


def test_failed_script_reports_failed(tmp_path):
    res = run_in_sandbox(["-c", "import sys; sys.exit(3)"], tmp_path, LIMITS)
    assert res.status == "failed"
    assert res.returncode == 3


def test_timeout_kills_process_and_reports_timeout(tmp_path):
    t0 = time.monotonic()
    res = run_in_sandbox(
        [str(SCRIPTS / "sleep.py")], tmp_path, SandboxLimits(1, 1000, 1)
    )
    assert res.status == "timeout"
    assert time.monotonic() - t0 < 5


def test_memory_limit_kills_process_and_reports_oom(tmp_path):
    res = run_in_sandbox(
        [str(SCRIPTS / "alloc.py")], tmp_path, SandboxLimits(30, 150, 1)
    )
    assert res.status == "oom"


def test_child_processes_are_killed_on_timeout(tmp_path):
    res = run_in_sandbox(
        [str(SCRIPTS / "sleep.py"), "spawn"], tmp_path, SandboxLimits(2, 1000, 1)
    )
    assert res.status == "timeout"
    pids = [int(p) for p in (tmp_path / "pids.txt").read_text().split()]
    assert pids
    time.sleep(0.3)
    assert not [p for p in pids if psutil.pid_exists(p) and psutil.Process(p).is_running()
                and psutil.Process(p).status() != psutil.STATUS_ZOMBIE]


def test_secrets_not_inherited(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_KEY", "supersecret123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "supersecret456")
    res = run_in_sandbox([str(SCRIPTS / "echo_env.py")], tmp_path, LIMITS)
    assert res.status == "ok"
    assert "supersecret" not in res.stdout
    assert "KAGGLE" not in res.stdout and "ANTHROPIC" not in res.stdout
    assert "OMP_NUM_THREADS=1" in res.stdout


def test_validate_operator_output_lists_missing_files(tmp_path):
    (tmp_path / "oof.npy").write_bytes(b"")
    with pytest.raises(SandboxOutputError) as exc:
        validate_operator_output(tmp_path)
    assert "test.npy" in str(exc.value)
    assert "metrics.json" in str(exc.value)
    assert "oof.npy" not in str(exc.value)


def test_validate_operator_output_returns_metrics(tmp_path):
    np.save(tmp_path / "oof.npy", np.zeros(3))
    np.save(tmp_path / "test.npy", np.zeros(2))
    metrics = {"cv_mean": 0.8, "cv_std": 0.01, "fold_scores": [0.8], "n_runs": 1}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    assert validate_operator_output(tmp_path) == metrics


def test_validate_operator_output_missing_metric_keys(tmp_path):
    np.save(tmp_path / "oof.npy", np.zeros(3))
    np.save(tmp_path / "test.npy", np.zeros(2))
    (tmp_path / "metrics.json").write_text(json.dumps({"cv_mean": 0.8}))
    with pytest.raises(SandboxOutputError) as exc:
        validate_operator_output(tmp_path)
    assert "cv_std" in str(exc.value)


def _alive(pid):
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait_dead(pids, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(_alive(p) for p in pids):
        time.sleep(0.1)
    return [p for p in pids if _alive(p)]


def test_base_exception_during_wait_kills_tree_and_reraises(tmp_path, monkeypatch):
    import subprocess

    started = []

    class InterruptedPopen(subprocess.Popen):
        def wait(self, timeout=None):
            started.append(self.pid)
            deadline = time.monotonic() + 15
            while not (tmp_path / "pids.txt").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            time.sleep(0.2)
            raise KeyboardInterrupt  # what Ctrl+C does to the orchestrator

    monkeypatch.setattr(subprocess, "Popen", InterruptedPopen)
    with pytest.raises(KeyboardInterrupt):
        run_in_sandbox([str(SCRIPTS / "sleep.py"), "spawn"], tmp_path, LIMITS)
    child = int((tmp_path / "pids.txt").read_text().split()[0])
    assert _wait_dead([started[0], child]) == []


def test_background_child_is_killed_after_normal_exit(tmp_path):
    res = run_in_sandbox([str(SCRIPTS / "orphan.py")], tmp_path, LIMITS)
    assert res.status == "ok"
    child = int((tmp_path / "pids.txt").read_text().split()[0])
    assert _wait_dead([child]) == []
