import pytest

from internet_experiment_lab.core import ExperimentEngine


def test_experiment_engine_discovers_presets():
    engine = ExperimentEngine()
    names = [experiment.name for experiment in engine.list_experiments()]
    assert "economy" in names
    assert "attention" in names
    assert "passwords" in names


def test_experiment_engine_run_returns_result():
    engine = ExperimentEngine()
    result = engine.run("economy", size=20, seed=1)
    assert result.dataset.shape[0] == 20
    assert result.metrics
    assert result.insight


def test_experiment_engine_invalid_name():
    engine = ExperimentEngine()
    with pytest.raises(ValueError, match="Unknown experiment"):
        engine.run("not_a_real_experiment", size=20, seed=1)
