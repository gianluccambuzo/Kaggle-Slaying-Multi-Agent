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


class CliKaggleClient:
    def __init__(
        self,
        runner: Callable[[list[str]], subprocess.CompletedProcess] = default_runner,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._runner = runner
        self._max_retries = max_retries
        self._sleep = sleep

    def _run(self, args: list[str]) -> str:
        cmd = ["kaggle", *args]
        last = ""
        for attempt in range(self._max_retries):
            res = self._runner(cmd)
            if res.returncode == 0:
                return res.stdout or ""
            last = f"{res.stdout or ''}\n{res.stderr or ''}".strip()
            if _RULES.search(last):
                raise KaggleRulesNotAccepted(last)
            if not _TRANSIENT.search(last):
                raise KaggleError(last)
            if attempt < self._max_retries - 1:
                self._sleep(2.0 ** attempt)
        raise KaggleTransientError(last)

    def download(self, competition: str, dest: Path) -> None:
        self._run(["competitions", "download", competition, "-p", str(dest), "--unzip"])

    def submit(self, competition: str, file: Path, message: str) -> None:
        self._run(
            ["competitions", "submit", competition, "-f", str(file), "-m", message, "--wait"]
        )

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
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
        csvs = sorted(dest.glob("*.csv"))
        if not csvs:
            raise KaggleError(f"no leaderboard CSV found in {dest}")
        return csvs[0]
