from dataclasses import dataclass
from pathlib import Path

from kagglers.core.types import PackRules


@dataclass(frozen=True)
class AirlineConfig:
    competition: str = "playground-series-s6e10"
    data_dir: Path = Path("data/playground-series-s6e10")
    positive_label: str | int | None = None
    n_splits: int = 5
    fold_seed: int = 42


RULES = PackRules(
    metric_name="roc_auc",
    higher_is_better=True,
    daily_submission_limit=5,
    allow_external_data=False,
    allow_test_derived_features=False,
)
