from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kagglers.core.types import Artifact, Score
from kagglers.packs.airline_s6e10.config import AirlineConfig


class SubmissionFormatError(Exception):
    pass


def validate_submission(df: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(df.columns) != list(sample.columns):
        raise SubmissionFormatError(
            f"columns {list(df.columns)} != expected {list(sample.columns)}"
        )
    if len(df) != len(sample):
        raise SubmissionFormatError(f"wrong row count: {len(df)} != {len(sample)}")
    id_col, target_col = sample.columns[0], sample.columns[1]
    if df[id_col].duplicated().any():
        raise SubmissionFormatError("duplicate ids")
    missing = set(sample[id_col]) - set(df[id_col])
    if missing:
        raise SubmissionFormatError(f"missing ids ({len(missing)})")
    if df[id_col].tolist() != sample[id_col].tolist():
        raise SubmissionFormatError("ids are not in sample_submission order")
    vals = pd.to_numeric(df[target_col], errors="coerce")
    if vals.isna().any():
        raise SubmissionFormatError("NaN or non-numeric values in prediction column")
    if ((vals < 0) | (vals > 1)).any():
        raise SubmissionFormatError("values outside range [0, 1]")


class AirlinePackager:
    def __init__(self, config: AirlineConfig):
        self.config = config

    def build(self, score: Score, out_dir: Path) -> Artifact:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        data_dir = Path(self.config.data_dir)
        sample = pd.read_csv(data_dir / "sample_submission.csv")
        id_col, target_col = sample.columns[0], sample.columns[1]
        # Predictions follow test.csv row order; match them to sample ids by id.
        test_ids = pd.read_csv(data_dir / "test.csv", usecols=[id_col])[id_col]
        pred = np.load(score.test_path)
        if pred.shape != (len(test_ids),):
            raise SubmissionFormatError(
                f"wrong row count: predictions {pred.shape} vs test.csv {len(test_ids)}"
            )
        if test_ids.duplicated().any():
            raise SubmissionFormatError("duplicate ids in test.csv")
        if sample[id_col].duplicated().any():
            raise SubmissionFormatError("duplicate ids in sample_submission.csv")
        missing = set(sample[id_col]) - set(test_ids)
        if missing:
            raise SubmissionFormatError(f"sample ids missing from test.csv ({len(missing)})")
        by_id = pd.Series(pred, index=test_ids.to_numpy())
        sub = pd.DataFrame(
            {id_col: sample[id_col], target_col: by_id.reindex(sample[id_col]).to_numpy()}
        )
        validate_submission(sub, sample)
        path = out_dir / "submission.csv"
        sub.to_csv(path, index=False)
        return Artifact(path=path, kind="submission")
