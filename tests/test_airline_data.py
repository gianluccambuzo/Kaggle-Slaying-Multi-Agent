import numpy as np
import pandas as pd
import pytest

from kagglers.packs.airline_s6e10.data import (
    Schema,
    adversarial_auc,
    encode_target,
    load_folds,
    load_train_test,
    make_folds,
    read_schema,
    save_folds,
    to_model_frame,
)


def test_read_schema_uses_first_two_sample_submission_columns(synthetic_comp):
    assert read_schema(synthetic_comp) == Schema(id_col="id", target_col="satisfaction")


def test_load_train_test_rejects_target_in_test(synthetic_comp):
    schema = read_schema(synthetic_comp)
    test = pd.read_csv(synthetic_comp / "test.csv")
    test["satisfaction"] = "satisfied"
    test.to_csv(synthetic_comp / "test.csv", index=False)
    with pytest.raises(ValueError, match="satisfaction"):
        load_train_test(synthetic_comp, schema)


def test_load_train_test_returns_frames(synthetic_comp):
    train, test = load_train_test(synthetic_comp, read_schema(synthetic_comp))
    assert len(train) == 600 and len(test) == 200
    assert "satisfaction" in train.columns and "satisfaction" not in test.columns


def test_encode_target_numeric_labels_need_no_positive_label():
    out = encode_target(pd.Series([0, 1, 1, 0]), None)
    assert out.tolist() == [0, 1, 1, 0]


def test_encode_target_text_labels_without_positive_label_raises_listing_values():
    y = pd.Series(["satisfied", "neutral or dissatisfied", "satisfied"])
    with pytest.raises(ValueError) as exc:
        encode_target(y, None)
    assert "satisfied" in str(exc.value)
    assert "neutral or dissatisfied" in str(exc.value)


def test_encode_target_maps_positive_label_to_one():
    y = pd.Series(["satisfied", "neutral or dissatisfied", "satisfied"])
    assert encode_target(y, "satisfied").tolist() == [1, 0, 1]


def _y():
    return np.random.default_rng(0).binomial(1, 0.3, 500)


def test_make_folds_is_deterministic_for_seed():
    y = _y()
    assert np.array_equal(make_folds(y, 5, 42), make_folds(y, 5, 42))


def test_make_folds_is_stratified():
    y = _y()
    folds = make_folds(y, 5, 42)
    assert set(folds.tolist()) == {0, 1, 2, 3, 4}
    for f in range(5):
        assert abs(y[folds == f].mean() - y.mean()) < 0.02


def test_save_load_folds_roundtrip(tmp_path):
    folds = make_folds(_y(), 5, 1)
    path = tmp_path / "folds.npy"
    save_folds(folds, path)
    assert np.array_equal(load_folds(path), folds)


def _frame(rng, n, shift=0.0):
    return pd.DataFrame(
        {
            "a": rng.normal(shift, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.choice(["x", "y"], n),
        }
    )


def test_adversarial_auc_near_half_for_same_distribution():
    rng = np.random.default_rng(0)
    assert adversarial_auc(_frame(rng, 400), _frame(rng, 400)) < 0.6


def test_adversarial_auc_high_for_shifted_distribution():
    rng = np.random.default_rng(0)
    assert adversarial_auc(_frame(rng, 400), _frame(rng, 400, shift=3.0)) > 0.8


def test_to_model_frame_handles_unseen_categories_and_nulls(synthetic_comp):
    train, test = load_train_test(synthetic_comp, read_schema(synthetic_comp))
    feats_tr = train.drop(columns=["id", "satisfaction"])
    feats_te = test.drop(columns=["id"])
    mtr, mte = to_model_frame(feats_tr, feats_te)
    assert list(mtr.columns) == list(mte.columns)
    assert mtr["travel_class"].dtype == "category"
    assert mtr["travel_class"].cat.categories.tolist() == mte["travel_class"].cat.categories.tolist()
    assert "First" in mte["travel_class"].cat.categories
    assert mte["travel_class"].isna().sum() == 0
    assert mte["distance"].isna().sum() == 10
    assert mte["customer_type"].isna().sum() == 5
