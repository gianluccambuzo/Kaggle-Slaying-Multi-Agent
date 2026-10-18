from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kagglers.core.sandbox import SandboxLimits, run_in_sandbox, validate_operator_output
from kagglers.core.types import Candidate, Score
from kagglers.packs.airline_s6e10.config import AirlineConfig
from kagglers.packs.airline_s6e10.data import load_train_test, read_schema

OPERATOR_MODULE = "kagglers.packs.airline_s6e10.baseline_operator"


class EvaluationFailed(Exception):
    def __init__(self, status: str, stderr: str):
        super().__init__(f"evaluation failed ({status}): {stderr[-2000:]}")
        self.status = status
        self.stderr = stderr


class AirlineEvaluator:
    def __init__(self, config: AirlineConfig, runs_dir: Path, limits: SandboxLimits):
        self.config = config
        self.runs_dir = Path(runs_dir)
        self.limits = limits

    def _folds_path(self) -> Path:
        c = self.config
        return Path(c.data_dir) / f"folds_k{c.n_splits}_s{c.fold_seed}.npy"

    def evaluate(self, candidate: Candidate) -> Score:
        run_dir = self.runs_dir / candidate.exp_id
        out_dir = run_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        params = candidate.params
        op_cfg = {
            "data_dir": str(Path(self.config.data_dir).resolve()),
            "folds_path": str(self._folds_path().resolve()),
            "out_dir": str(out_dir.resolve()),
            "positive_label": self.config.positive_label,
            "seeds": params.get("seeds", [0]),
            "params": params.get("lgbm", {}),
            "n_jobs": self.limits.cpu_threads,
        }
        cfg_path = run_dir / "operator_config.json"
        cfg_path.write_text(json.dumps(op_cfg), encoding="utf-8")
        result = run_in_sandbox(["-m", OPERATOR_MODULE, str(cfg_path)], run_dir, self.limits)
        if result.status != "ok":
            raise EvaluationFailed(result.status, result.stderr)
        metrics = validate_operator_output(out_dir)

        schema = read_schema(self.config.data_dir)
        train, test = load_train_test(self.config.data_dir, schema)
        oof = np.load(out_dir / "oof.npy")
        pred = np.load(out_dir / "test.npy")
        if oof.shape != (len(train),):
            raise EvaluationFailed("bad_output", f"oof shape {oof.shape} != ({len(train)},)")
        if pred.shape != (len(test),):
            raise EvaluationFailed("bad_output", f"test shape {pred.shape} != ({len(test)},)")
        return Score(
            value=float(metrics["cv_mean"]),
            std=float(metrics["cv_std"]),
            n_runs=int(metrics["n_runs"]),
            fold_scores=[float(s) for s in metrics["fold_scores"]],
            oof_path=out_dir / "oof.npy",
            test_path=out_dir / "test.npy",
        )
