from kagglers.cli import main
from kagglers.core.store import Store
from tests.fakes import FakeKaggleClient

PACK_KWARGS = {"positive_label": "satisfied"}


def test_cli_parses_run_command_and_returns_zero_on_success(synthetic_comp, tmp_path, capsys):
    base = tmp_path / "base"
    kaggle = FakeKaggleClient(fixture_dir=synthetic_comp)

    code = main(
        ["run", "airline_s6e10", "--phase1", "--compute-minutes", "30"],
        kaggle=kaggle,
        base_dir=base,
        pack_kwargs=PACK_KWARGS,
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "CV roc_auc:" in out and "±" in out
    assert "submission.csv" in out
    assert "kaggle_ref: 1" in out
    assert "percentile: top 10.0%" in out
    assert "stop_reason: phase1_complete" in out
    assert len(kaggle.submitted) == 1
    assert (base / "runs" / "kagglers.db").exists()
    assert (base / "data" / "playground-series-s6e10" / "train.csv").exists()


def test_cli_resume_returns_nonzero_then_zero(synthetic_comp, tmp_path, capsys):
    base = tmp_path / "base"
    kaggle = FakeKaggleClient(fixture_dir=synthetic_comp, fail_leaderboard_once=True)
    assert main(["run", "airline_s6e10", "--phase1"], kaggle=kaggle, base_dir=base,
                pack_kwargs=PACK_KWARGS) == 1
    out = capsys.readouterr().out
    assert "stop_reason: error: KaggleTransientError" in out

    (run_id,) = [ln.split(":", 1)[1].strip() for ln in out.splitlines()
                 if ln.startswith("run_id:")]
    code = main(["run", "airline_s6e10", "--phase1", "--resume", run_id],
                kaggle=kaggle, base_dir=base, pack_kwargs=PACK_KWARGS)
    assert code == 0
    assert len(kaggle.submitted) == 1
    store = Store(base / "runs" / "kagglers.db")
    assert len(store.list_experiments(run_id)) == 1
    store.close()


def test_cli_unknown_pack_returns_nonzero_with_message(tmp_path, capsys):
    code = main(["run", "nope", "--phase1"], kaggle=FakeKaggleClient(), base_dir=tmp_path)
    err = capsys.readouterr().err
    assert code != 0
    assert "unknown pack 'nope'" in err
    assert "airline_s6e10" in err


def test_cli_unknown_resume_id_returns_nonzero(tmp_path, capsys):
    code = main(["run", "airline_s6e10", "--phase1", "--resume", "deadbeef"],
                kaggle=FakeKaggleClient(), base_dir=tmp_path)
    assert code != 0
    assert "deadbeef" in capsys.readouterr().err


def test_cli_requires_phase1_flag(tmp_path, capsys):
    code = main(["run", "airline_s6e10"], kaggle=FakeKaggleClient(), base_dir=tmp_path)
    assert code != 0
    assert "--phase1" in capsys.readouterr().err
