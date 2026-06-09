import pytest

from internet_experiment_lab.custom_model import CustomModelRunner, default_custom_spec


def test_custom_model_default_spec_runs():
    run = CustomModelRunner().run(default_custom_spec(), size=100, seed=42)
    assert run.result.dataset.shape[0] == 100
    assert "reach_score" in run.result.dataset.columns
    assert "average_reach_score" in run.result.metrics
    assert run.result.insight
    assert run.result.title == "Creator Growth Experiment"


def test_custom_model_invalid_variable_type():
    spec = default_custom_spec()
    spec["variables"][0]["type"] = "unknown_type"
    with pytest.raises(ValueError, match="Unsupported variable type"):
        CustomModelRunner().run(spec, size=100, seed=1)


def test_custom_model_duplicate_variable_names():
    spec = default_custom_spec()
    spec["variables"][1]["name"] = spec["variables"][0]["name"]
    with pytest.raises(ValueError, match="Duplicate variable name"):
        CustomModelRunner().run(spec, size=100, seed=1)


def test_custom_model_unknown_formula_name():
    spec = default_custom_spec()
    spec["derived"][0]["formula"] = "log(unknown_field)"
    with pytest.raises(ValueError, match="Formula references unknown name"):
        CustomModelRunner().run(spec, size=100, seed=1)
