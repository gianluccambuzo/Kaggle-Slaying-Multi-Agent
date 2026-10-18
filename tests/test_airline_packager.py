import numpy as np
import pandas as pd
import pytest

from kagglers.core.types import Score
from kagglers.packs.airline_s6e10.config import AirlineConfig
from kagglers.packs.airline_s6e10.packager import (
    AirlinePackager,
    SubmissionFormatError,
    validate_submission,
)


@pytest.fixture
def sample():
    return pd.DataFrame({"id": [10, 11, 12, 13], "satisfaction": "satisfied"})


def _good(sample):
    return pd.DataFrame({"id": sample["id"], "satisfaction": [0.1, 0.5, 0.9, 0.3]})


def test_build_writes_csv_matching_sample_submission_columns_and_order(synthetic_comp, tmp_path):
    test_pred = np.linspace(0.05, 0.95, 200)
    np.save(tmp_path / "test.npy", test_pred)
    score = Score(0.9, 0.0, 5, [0.9] * 5, tmp_path / "oof.npy", tmp_path / "test.npy")
    cfg = AirlineConfig(data_dir=synthetic_comp, positive_label="satisfied")
    art = AirlinePackager(cfg).build(score, tmp_path / "out")
    sub = pd.read_csv(art.path)
    sample = pd.read_csv(synthetic_comp / "sample_submission.csv")
    assert list(sub.columns) == list(sample.columns)
    assert sub["id"].tolist() == sample["id"].tolist()
    assert np.allclose(sub["satisfaction"], test_pred)
    assert art.kind == "submission"


def test_build_has_no_nan_and_probabilities_in_unit_interval(synthetic_comp, tmp_path):
    np.save(tmp_path / "test.npy", np.random.default_rng(0).random(200))
    score = Score(0.9, 0.0, 5, [0.9] * 5, tmp_path / "oof.npy", tmp_path / "test.npy")
    cfg = AirlineConfig(data_dir=synthetic_comp, positive_label="satisfied")
    sub = pd.read_csv(AirlinePackager(cfg).build(score, tmp_path / "out").path)
    assert not sub["satisfaction"].isna().any()
    assert sub["satisfaction"].between(0, 1).all()


def test_validate_submission_accepts_good(sample):
    validate_submission(_good(sample), sample)


def test_validate_submission_rejects_missing_ids(sample):
    df = _good(sample)
    df.loc[1, "id"] = 99
    with pytest.raises(SubmissionFormatError, match="missing"):
        validate_submission(df, sample)


def test_validate_submission_rejects_duplicate_ids(sample):
    df = _good(sample)
    df.loc[3, "id"] = 10
    with pytest.raises(SubmissionFormatError, match="duplicate"):
        validate_submission(df, sample)


def test_validate_submission_rejects_reordered_ids(sample):
    df = _good(sample).iloc[::-1].reset_index(drop=True)
    with pytest.raises(SubmissionFormatError, match="order"):
        validate_submission(df, sample)


def test_validate_submission_rejects_nan(sample):
    df = _good(sample)
    df.loc[2, "satisfaction"] = np.nan
    with pytest.raises(SubmissionFormatError, match="NaN"):
        validate_submission(df, sample)


def test_validate_submission_rejects_wrong_row_count(sample):
    with pytest.raises(SubmissionFormatError, match="row count"):
        validate_submission(_good(sample).iloc[:3], sample)


def test_validate_submission_rejects_out_of_range(sample):
    df = _good(sample)
    df.loc[0, "satisfaction"] = 1.5
    with pytest.raises(SubmissionFormatError, match="range"):
        validate_submission(df, sample)
