from __future__ import annotations

import shutil
from pathlib import Path

from kagglers.core.kaggle_client import (
    KaggleError,
    KaggleRulesNotAccepted,
    KaggleTransientError,
    SubmissionInfo,
)


class FakeKaggleClient:
    def __init__(
        self,
        fixture_dir: Path | None = None,
        leaderboard_scores: list[float] | None = None,
        public_score: float = 0.95,
        fail_submit_after_success: bool = False,
        fail_leaderboard_once: bool = False,
        rules_not_accepted: bool = False,
    ) -> None:
        self.fixture_dir = fixture_dir
        self.leaderboard_scores = leaderboard_scores if leaderboard_scores is not None else [
            0.5 + i * 0.01 for i in range(10)
        ]
        self.public_score = public_score
        self.fail_submit_after_success = fail_submit_after_success
        self.fail_leaderboard_once = fail_leaderboard_once
        self.rules_not_accepted = rules_not_accepted
        self.submitted: list[tuple[str, Path, str, str]] = []
        self.leaderboard_calls = 0

    def download(self, competition: str, dest: Path) -> None:
        if self.rules_not_accepted:
            raise KaggleRulesNotAccepted("403 - you must accept the rules")
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        if self.fixture_dir is not None:
            for f in Path(self.fixture_dir).iterdir():
                if f.is_file():
                    shutil.copy(f, dest / f.name)

    def submit(self, competition: str, file: Path, message: str) -> None:
        ref = str(len(self.submitted) + 1)
        self.submitted.append((competition, Path(file), message, ref))
        if self.fail_submit_after_success:
            raise KaggleError("crash after CLI success")

    def submissions(self, competition: str) -> list[SubmissionInfo]:
        return [
            SubmissionInfo(ref, msg, "complete", self.public_score)
            for comp, _f, msg, ref in self.submitted
            if comp == competition
        ]

    def leaderboard(self, competition: str, dest: Path) -> Path:
        self.leaderboard_calls += 1
        if self.fail_leaderboard_once and self.leaderboard_calls == 1:
            raise KaggleTransientError("503 unavailable")
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "leaderboard.csv"
        lines = ["TeamId,TeamName,SubmissionDate,Score"]
        for i, s in enumerate(self.leaderboard_scores):
            lines.append(f"{i},team{i},2026-01-01 00:00:00,{s}")
        path.write_text("\n".join(lines) + "\n")
        return path
