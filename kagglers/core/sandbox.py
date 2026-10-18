"""Subprocess sandbox for operator code: wall-clock and RSS limits, minimal env."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

POLL_INTERVAL_S = 0.2
REQUIRED_FILES = ("oof.npy", "test.npy", "metrics.json")
REQUIRED_METRIC_KEYS = ("cv_mean", "cv_std", "fold_scores", "n_runs")


class SandboxOutputError(Exception):
    """Operator output is missing files or metric keys."""


@dataclass(frozen=True)
class SandboxLimits:
    timeout_s: float
    memory_mb: int
    cpu_threads: int


@dataclass(frozen=True)
class SandboxResult:
    status: str  # "ok" | "failed" | "timeout" | "oom"
    returncode: int | None
    duration_s: float
    stdout: str
    stderr: str


def _minimal_env(cpu_threads: int) -> dict[str, str]:
    env = {"OMP_NUM_THREADS": str(cpu_threads)}
    # SYSTEMROOT is required on Windows for the interpreter to start (winsock/crypto
    # init); TEMP/TMP so tempfile works. Nothing else is inherited (no secrets).
    for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _tree(proc: psutil.Process) -> list[psutil.Process]:
    try:
        return proc.children(recursive=True) + [proc]
    except psutil.NoSuchProcess:
        return []


def _kill_procs(procs: list[psutil.Process]) -> None:
    alive = []
    for p in procs:
        try:
            if p.is_running():  # False if the pid was reused by another process
                p.kill()
                alive.append(p)
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=5)


def _kill_tree(proc: psutil.Process) -> None:
    _kill_procs(_tree(proc))  # children first, parent last


def _tree_rss_mb(proc: psutil.Process) -> float:
    total = 0
    for p in _tree(proc):
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / (1024 * 1024)


def run_in_sandbox(args: list[str], workdir: Path, limits: SandboxLimits) -> SandboxResult:
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    start = time.monotonic()
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        popen = subprocess.Popen(
            [sys.executable, "-I", *args],
            cwd=str(workdir),
            env=_minimal_env(limits.cpu_threads),
            stdin=subprocess.DEVNULL,
            stdout=out_f,
            stderr=err_f,
            creationflags=flags,
        )
        ps = psutil.Process(popen.pid)
        verdict: list[str] = []
        done = threading.Event()
        # Descendants seen while the child ran: once it exits they can no longer be
        # found through it, and leftovers must not outlive the sandbox.
        seen: dict[int, psutil.Process] = {}

        def snapshot() -> None:
            try:
                for c in ps.children(recursive=True):
                    seen.setdefault(c.pid, c)
            except psutil.Error:
                pass

        def monitor() -> None:
            while not done.wait(POLL_INTERVAL_S):
                snapshot()
                if time.monotonic() - start > limits.timeout_s:
                    verdict.append("timeout")
                elif _tree_rss_mb(ps) > limits.memory_mb:
                    verdict.append("oom")
                else:
                    continue
                _kill_tree(ps)
                return

        t = threading.Thread(target=monitor, daemon=True)
        t.start()
        try:
            popen.wait()
        except BaseException:  # Ctrl+C or any abort: never leave the tree running
            done.set()
            t.join()
            snapshot()
            _kill_tree(ps)
            _kill_procs(list(seen.values()))
            popen.poll()
            raise
        finally:
            done.set()
            t.join()
        _kill_procs(list(seen.values()))
        returncode = popen.returncode
        out_f.seek(0)
        err_f.seek(0)
        stdout = out_f.read().decode("utf-8", errors="replace")
        stderr = err_f.read().decode("utf-8", errors="replace")
    duration = time.monotonic() - start
    if verdict:
        status = verdict[0]
    else:
        status = "ok" if returncode == 0 else "failed"
    return SandboxResult(status, returncode, duration, stdout, stderr)


def validate_operator_output(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    missing = [f for f in REQUIRED_FILES if not (out_dir / f).is_file()]
    if missing:
        raise SandboxOutputError(f"missing operator output files: {missing}")
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    missing_keys = [k for k in REQUIRED_METRIC_KEYS if k not in metrics]
    if missing_keys:
        raise SandboxOutputError(f"missing metrics keys: {missing_keys}")
    return metrics
