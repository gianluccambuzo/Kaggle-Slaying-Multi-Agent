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


def test_get_pack_delegates_to_build_pack(tmp_path, monkeypatch):
    import kagglers.packs.airline_s6e10 as airline
    from kagglers.core.sandbox import SandboxLimits
    from kagglers.packs import get_pack

    calls = []
    real = airline.build_pack

    def spy(config, workdir, limits=None):
        calls.append((config, workdir, limits))
        return real(config, workdir, limits)

    monkeypatch.setattr(airline, "build_pack", spy)
    limits = SandboxLimits(timeout_s=5, memory_mb=100, cpu_threads=1)
    pack = get_pack("airline_s6e10", workdir=tmp_path, positive_label="satisfied", limits=limits)

    (config, workdir, got_limits) = calls[0]
    assert workdir == tmp_path and got_limits is limits
    assert config.positive_label == "satisfied"
    assert pack.data_dir == tmp_path / "data" / "playground-series-s6e10"
    assert pack.evaluator.limits is limits
    assert not hasattr(airline, "PACKS")
