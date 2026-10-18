from __future__ import annotations

import csv
import io
import re
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


class KaggleError(Exception):
    pass


class KaggleRulesNotAccepted(KaggleError):
    pass


class KaggleTransientError(KaggleError):
    pass


@dataclass(frozen=True)
class SubmissionInfo:
    ref: str
    description: str
    status: str
    public_score: float | None


class KaggleClient(Protocol):
    def download(self, competition: str, dest: Path) -> None: ...

    def submit(self, competition: str, file: Path, message: str) -> None: ...

    def submissions(self, competition: str) -> list[SubmissionInfo]: ...

    def leaderboard(self, competition: str, dest: Path) -> Path: ...


def default_runner(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


_RULES = re.compile(r"\b403\b|accept.*rules|rules.*accept", re.I)
_TRANSIENT = re.compile(r"\b(5\d\d|429)\b|timed? ?out|connection|temporar|unavailable", re.I)


DEFAULT_MAX_POLLS = 60
DEFAULT_POLL_INTERVAL_S = 10.0


def classify_status(raw: str) -> str:
    """Map a CLI status (`SubmissionStatus.COMPLETE`, `COMPLETE`, `ERROR`, `PENDING`,
    ...) to `complete`, `error` or `pending`."""
    s = (raw or "").lower()
    if "complete" in s:
        return "complete"
    if "error" in s or "fail" in s:
        return "error"
    return "pending"


def _safe_extract(zip_path: Path, dest: Path) -> None:
    root = Path(dest).resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            target = (root / name).resolve()
            if target != root and root not in target.parents:
                raise KaggleError(f"unsafe path {name!r} in {zip_path}")
        zf.extractall(root)


class CliKaggleClient:
    def __init__(
        self,
        runner: Callable[[list[str]], subprocess.CompletedProcess] = default_runner,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        max_polls: int = DEFAULT_MAX_POLLS,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        self._runner = runner
        self._max_retries = max_retries
        self._sleep = sleep
        self.max_polls = max_polls
        self.poll_interval_s = poll_interval_s

    def _run_once(self, args: list[str]) -> str:
        res = self._runner(["kaggle", *args])
        if res.returncode == 0:
            return res.stdout or ""
        text = f"{res.stdout or ''}\n{res.stderr or ''}".strip()
        if _RULES.search(text):
            raise KaggleRulesNotAccepted(text)
        if not _TRANSIENT.search(text):
            raise KaggleError(text)
        raise KaggleTransientError(text)

    def _run(self, args: list[str]) -> str:
        """Run with retry and exponential backoff; only for idempotent commands."""
        for attempt in range(self._max_retries):
            try:
                return self._run_once(args)
            except KaggleTransientError:
                if attempt == self._max_retries - 1:
                    raise
                self._sleep(2.0 ** attempt)
        raise AssertionError("unreachable")

    def download(self, competition: str, dest: Path) -> None:
        # The CLI (2.x) writes <dest>/<competition>.zip and has no --unzip flag.
        dest = Path(dest)
        self._run(["competitions", "download", competition, "-p", str(dest)])
        archive = dest / f"{competition}.zip"
        if not archive.is_file():
            raise KaggleError(f"download finished but {archive} was not found")
        _safe_extract(archive, dest)

    def submit(self, competition: str, file: Path, message: str) -> None:
        """Upload once (never retried: a retry could double-submit), then poll
        `submissions()` until the entry with description == message is terminal or
        `max_polls` is reached; a still-pending entry is left to the caller."""
        self._run_once(["competitions", "submit", competition, "-f", str(file), "-m", message])
        self._wait_for_status(competition, message)

    def _wait_for_status(self, competition: str, message: str) -> None:
        for poll in range(self.max_polls):
            info = next(
                (s for s in self.submissions(competition) if s.description == message), None
            )
            if info is not None and classify_status(info.status) != "pending":
                return
            if poll < self.max_polls - 1:
                self._sleep(self.poll_interval_s)

    def submissions(self, competition: str) -> list[SubmissionInfo]:
        out = self._run(["competitions", "submissions", competition, "-v"])
        # NOTE: column names unverified against the real CLI; parse by name.
        rows = csv.DictReader(io.StringIO(out))
        result = []
        for row in rows:
            raw = (row.get("publicScore") or "").strip()
            try:
                score = float(raw) if raw else None
            except ValueError:
                score = None
            result.append(
                SubmissionInfo(
                    ref=(row.get("ref") or "").strip(),
                    description=(row.get("description") or "").strip(),
                    status=(row.get("status") or "").strip(),
                    public_score=score,
                )
            )
        return result

    def leaderboard(self, competition: str, dest: Path) -> Path:
        dest = Path(dest)
        self._run(["competitions", "leaderboard", competition, "-d", "-p", str(dest)])
        for z in dest.glob("*.zip"):
            _safe_extract(z, dest)
        csvs = sorted(dest.glob("*.csv"))
        if not csvs:
            raise KaggleError(f"no leaderboard CSV found in {dest}")
        return csvs[0]

