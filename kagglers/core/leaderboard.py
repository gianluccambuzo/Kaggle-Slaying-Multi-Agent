from __future__ import annotations

from pathlib import Path

import pandas as pd


def top_percent(our_score: float, leaderboard_csv: Path, higher_is_better: bool = True) -> float:
    df = pd.read_csv(leaderboard_csv)
    n_teams = len(df)
    if n_teams == 0:
        raise ValueError("empty leaderboard")
    scores = pd.to_numeric(df["Score"], errors="coerce")
    better = scores > our_score if higher_is_better else scores < our_score
    return 100.0 * (int(better.sum()) + 1) / n_teams
