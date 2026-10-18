from __future__ import annotations

from pathlib import Path

from kagglers.core.sandbox import SandboxLimits
from kagglers.packs.airline_s6e10.config import RULES, AirlineConfig
from kagglers.packs.airline_s6e10.data import (
    adversarial_auc,
    encode_target,
    load_train_test,
    make_folds,
    read_schema,
    save_folds,
)
from kagglers.packs.airline_s6e10.evaluator import AirlineEvaluator
from kagglers.packs.airline_s6e10.packager import AirlinePackager

NAME = "airline_s6e10"

DEFAULT_LIMITS = SandboxLimits(timeout_s=3600, memory_mb=8000, cpu_threads=4)


def default_baseline_params() -> dict:
    return {
        "seeds": [0],
        "lgbm": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "verbosity": -1,
        },
    }


class AirlinePack:
    name = NAME

    def __init__(
        self, config: AirlineConfig, workdir: Path, limits: SandboxLimits = DEFAULT_LIMITS
    ):
        self.config = config
        self.workdir = Path(workdir)
        self.competition = config.competition
        self.rules = RULES
        self.data_dir = Path(config.data_dir)
        self.baseline_params = default_baseline_params()
        self.evaluator = AirlineEvaluator(config, self.workdir / "runs", limits)
        self.packager = AirlinePackager(config)
        self.adversarial_auc: float | None = None

    def prepare(self) -> None:
        c = self.config
        schema = read_schema(self.data_dir)
        train, test = load_train_test(self.data_dir, schema)
        y = encode_target(train[schema.target_col], c.positive_label)
        folds = make_folds(y, c.n_splits, c.fold_seed)
        save_folds(folds, self.data_dir / f"folds_k{c.n_splits}_s{c.fold_seed}.npy")
        feats = [col for col in train.columns if col not in (schema.id_col, schema.target_col)]
        self.adversarial_auc = adversarial_auc(train[feats], test[feats])
