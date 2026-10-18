from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Hypothesis:
    text: str
    operator: str
    params: dict


@dataclass(frozen=True)
class Candidate:
    exp_id: str
    operator: str
    params: dict

    @property
    def config_hash(self) -> str:
        canonical = json.dumps(
            {"operator": self.operator, "params": self.params},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Score:
    value: float
    std: float
    n_runs: int
    fold_scores: list[float]
    oof_path: Path
    test_path: Path


@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: str


@dataclass(frozen=True)
class PackRules:
    metric_name: str
    higher_is_better: bool
    daily_submission_limit: int
    allow_external_data: bool
    allow_test_derived_features: bool


class Evaluator(Protocol):
    def evaluate(self, candidate: Candidate) -> Score: ...


class Packager(Protocol):
    def build(self, score: Score, out_dir: Path) -> Artifact: ...


class Pack(Protocol):
    name: str
    competition: str
    rules: PackRules
    evaluator: Evaluator
    packager: Packager
    data_dir: Path
    baseline_params: dict

    def prepare(self) -> None: ...
