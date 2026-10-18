import numpy as np

from kagglers.packs.airline_s6e10 import build_pack
from kagglers.packs.airline_s6e10.config import AirlineConfig
from kagglers.packs.airline_s6e10.data import load_folds


def test_prepare_writes_folds_file_and_sets_adversarial_auc(synthetic_comp, tmp_path):
    cfg = AirlineConfig(data_dir=synthetic_comp, positive_label="satisfied")
    pack = build_pack(cfg, tmp_path / "work")
    pack.prepare()
    folds = load_folds(pack.data_dir / "folds_k5_s42.npy")
    assert folds.shape == (600,)
    assert set(np.unique(folds)) == {0, 1, 2, 3, 4}
    assert 0.0 <= pack.adversarial_auc <= 1.0
    assert pack.name == "airline_s6e10"
    assert pack.baseline_params["seeds"] == [0]
    assert pack.baseline_params["lgbm"]["verbosity"] == -1
