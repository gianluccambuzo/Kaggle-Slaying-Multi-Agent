from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _features(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "distance": rng.normal(1000, 400, n).round(1),
            "delay": rng.exponential(15, n).round(1),
            "service": rng.integers(1, 6, n),
            "travel_class": rng.choice(["Eco", "Business"], n),
            "customer_type": rng.choice(["loyal", "disloyal"], n),
        }
    )


@pytest.fixture
def synthetic_comp(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1234)
    train = _features(rng, 600)
    score = (
        0.8 * (train["service"] - 3)
        - 0.03 * train["delay"]
        + 0.8 * (train["travel_class"] == "Business")
        + rng.normal(0, 0.5, 600)
    )
    train.insert(0, "id", np.arange(600))
    train["satisfaction"] = np.where(score > 0, "satisfied", "neutral or dissatisfied")

    test = _features(rng, 200)
    test.insert(0, "id", np.arange(600, 800))
    test.loc[test.index[:10], "travel_class"] = "First"  # unseen in train
    test.loc[test.index[10:20], "distance"] = np.nan
    test.loc[test.index[20:25], "customer_type"] = None

    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "satisfaction": "satisfied"}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )
    return tmp_path
