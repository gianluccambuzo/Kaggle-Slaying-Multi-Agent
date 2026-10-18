"""Baseline operator: LightGBM over pre-made folds. Run via `python -I -m ... <config.json>`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from kagglers.packs.airline_s6e10.data import (
    encode_target,
    load_folds,
    load_train_test,
    read_schema,
    to_model_frame,
)


def main(config_path: Path) -> None:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["out_dir"])
    seeds = [int(s) for s in cfg["seeds"]]
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = read_schema(data_dir)
    train, test = load_train_test(data_dir, schema)
    y = encode_target(train[schema.target_col], cfg["positive_label"])
    folds = load_folds(Path(cfg["folds_path"]))
    if folds.shape != (len(train),):
        raise ValueError(f"folds shape {folds.shape} does not match n_train={len(train)}")
    X, X_test = to_model_frame(
        train.drop(columns=[schema.id_col, schema.target_col]),
        test.drop(columns=[schema.id_col]),
    )
    fold_ids = sorted(int(f) for f in np.unique(folds))

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    for seed in seeds:
        for f in fold_ids:
            val = folds == f
            model = lgb.LGBMClassifier(
                **cfg["params"], random_state=seed, n_jobs=int(cfg["n_jobs"])
            )
            model.fit(X[~val], y[~val])
            oof[val] += model.predict_proba(X[val])[:, 1] / len(seeds)
            test_pred += model.predict_proba(X_test)[:, 1] / (len(seeds) * len(fold_ids))

    fold_scores = [float(roc_auc_score(y[folds == f], oof[folds == f])) for f in fold_ids]
    np.save(out_dir / "oof.npy", oof)
    np.save(out_dir / "test.npy", test_pred)
    metrics = {
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
        "n_runs": len(seeds) * len(fold_ids),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]))
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.exit(1)
