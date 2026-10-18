from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class Schema:
    id_col: str
    target_col: str


def read_schema(data_dir: Path) -> Schema:
    cols = list(pd.read_csv(Path(data_dir) / "sample_submission.csv", nrows=0).columns)
    if len(cols) < 2:
        raise ValueError("sample_submission.csv must have at least two columns")
    return Schema(id_col=cols[0], target_col=cols[1])


def load_train_test(data_dir: Path, schema: Schema) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    for col in (schema.id_col, schema.target_col):
        if col not in train.columns:
            raise ValueError(f"train.csv is missing column {col!r}")
    if schema.id_col not in test.columns:
        raise ValueError(f"test.csv is missing column {schema.id_col!r}")
    if schema.target_col in test.columns:
        raise ValueError(f"test.csv must not contain target column {schema.target_col!r}")
    train_feats = [c for c in train.columns if c not in (schema.id_col, schema.target_col)]
    test_feats = [c for c in test.columns if c != schema.id_col]
    if set(train_feats) != set(test_feats):
        raise ValueError(
            f"feature columns differ: train-only={sorted(set(train_feats) - set(test_feats))}, "
            f"test-only={sorted(set(test_feats) - set(train_feats))}"
        )
    return train, test


def encode_target(y: pd.Series, positive_label: str | int | None) -> np.ndarray:
    if positive_label is None:
        uniques = set(y.dropna().unique().tolist())
        if uniques <= {0, 1}:
            return y.to_numpy().astype(int)
        raise ValueError(
            f"positive_label is required for non-binary-numeric target; unique values: "
            f"{sorted(uniques, key=str)}"
        )
    return (y == positive_label).to_numpy().astype(int)


def make_folds(y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    folds = np.full(len(y), -1, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[val_idx] = fold
    return folds


def save_folds(folds: np.ndarray, path: Path) -> None:
    np.save(Path(path), folds)


def load_folds(path: Path) -> np.ndarray:
    return np.load(Path(path))


def to_model_frame(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert text columns to category dtype using the train+test union of categories."""
    train, test = train.copy(), test.copy()
    for col in train.columns:
        if col not in test.columns:
            continue
        if not (pd.api.types.is_numeric_dtype(train[col]) and pd.api.types.is_numeric_dtype(test[col])):
            if pd.api.types.is_bool_dtype(train[col]) and pd.api.types.is_bool_dtype(test[col]):
                continue
            cats = pd.concat([train[col], test[col]]).dropna().unique().tolist()
            cats = sorted(cats, key=str)
            dtype = pd.CategoricalDtype(categories=cats)
            train[col] = train[col].astype(dtype)
            test[col] = test[col].astype(dtype)
    return train, test


def adversarial_auc(
    train: pd.DataFrame, test: pd.DataFrame, sample: int = 50_000, seed: int = 0
) -> float:
    """CV AUC of a small model separating train from test rows (0.5 = same distribution)."""
    n = min(sample, len(train), len(test))
    tr = train.sample(n=n, random_state=seed)
    te = test.sample(n=n, random_state=seed)
    tr, te = to_model_frame(tr, te)
    X = pd.concat([tr, te], ignore_index=True)
    y = np.r_[np.zeros(len(tr), dtype=int), np.ones(len(te), dtype=int)]
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for fit_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(
            n_estimators=50,
            learning_rate=0.1,
            num_leaves=15,
            min_child_samples=10,
            n_jobs=1,
            random_state=seed,
            verbosity=-1,
        )
        model.fit(X.iloc[fit_idx], y[fit_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return float(roc_auc_score(y, oof))
