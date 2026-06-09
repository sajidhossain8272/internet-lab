from __future__ import annotations

import ast
import functools
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from internet_experiment_lab.core import ChartSpec, ExperimentResult
from internet_experiment_lab.math_utils import correlation, sigmoid


SAFE_FUNCTIONS = {
    "abs": np.abs,
    "clip": np.clip,
    "exp": np.exp,
    "log": np.log,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "round": np.round,
    "sigmoid": sigmoid,
    "sqrt": np.sqrt,
    "where": np.where,
}

SUPPORTED_VARIABLE_TYPES = {"normal", "uniform", "lognormal", "beta", "poisson", "bernoulli", "categorical"}
SUPPORTED_METRIC_OPS = {"mean", "median", "std", "min", "max", "rate", "count", "unique", "corr"}
SUPPORTED_CHART_KINDS = {"histogram", "scatter", "bar"}

ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.BitAnd,
    ast.BitOr,
)


def _spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=32)
def _compile_formula(formula: str, allowed_columns: tuple[str, ...]) -> Any:
    tree = ast.parse(formula, mode="eval")
    allowed_names = set(allowed_columns) | set(SAFE_FUNCTIONS)
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise ValueError(f"Formula uses unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Formula references unknown name: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
                raise ValueError("Formula can only call safe functions.")
    return compile(tree, "<custom_model_formula>", "eval")


def _sample_variable_value(variable: dict[str, Any], size: int, rng: np.random.Generator) -> Any:
    kind = str(variable.get("type", "normal")).lower()
    if kind == "normal":
        return rng.normal(_float(variable, "mean", 0), _float(variable, "sd", 1), size=size)
    if kind == "uniform":
        return rng.uniform(_float(variable, "min", 0), _float(variable, "max", 1), size=size)
    if kind == "lognormal":
        return rng.lognormal(_float(variable, "mean", 1), _float(variable, "sigma", 0.5), size=size)
    if kind == "beta":
        return rng.beta(_float(variable, "a", 2), _float(variable, "b", 2), size=size)
    if kind == "poisson":
        return rng.poisson(max(_float(variable, "lambda", 1), 0.01), size=size)
    if kind == "bernoulli":
        return rng.random(size) < min(max(_float(variable, "p", 0.5), 0), 1)
    if kind == "categorical":
        choices = variable.get("choices", ["a", "b"])
        if not isinstance(choices, list) or not choices:
            raise ValueError("Categorical variables need a non-empty choices list.")
        probabilities = variable.get("probabilities")
        if probabilities is not None:
            probabilities = np.asarray(probabilities, dtype=float)
            probabilities = probabilities / probabilities.sum()
        return rng.choice(np.asarray(choices, dtype=str), size=size, p=probabilities)
    raise ValueError(f"Unsupported variable type: {kind}")


@functools.lru_cache(maxsize=16)
def _build_variable_dataset(variables_json: str, size: int, seed: int) -> pd.DataFrame:
    variables = json.loads(variables_json)
    rng = np.random.default_rng(seed)
    data: dict[str, Any] = {}
    for variable in variables:
        name = _slug(variable.get("name", "field"))
        if not name:
            raise ValueError("Every variable needs a name.")
        data[name] = _sample_variable_value(variable, size, rng)
    return pd.DataFrame(data)


@dataclass(frozen=True)
class CustomRun:
    result: ExperimentResult
    design: dict[str, Any]


def default_custom_spec() -> dict[str, Any]:
    return {
        "name": "creator_growth",
        "title": "Creator Growth Experiment",
        "hypothesis": "A creator's posting consistency and novelty should raise reach, but burnout risk grows with posting pressure.",
        "variables": [
            {"name": "followers", "type": "lognormal", "mean": 8.2, "sigma": 1.1},
            {"name": "posting_days", "type": "poisson", "lambda": 4.2},
            {"name": "novelty", "type": "beta", "a": 2.4, "b": 3.1},
            {"name": "hook_quality", "type": "normal", "mean": 0.62, "sd": 0.16},
            {"name": "uses_thread", "type": "bernoulli", "p": 0.35},
            {"name": "niche", "type": "categorical", "choices": ["ai", "money", "health", "career"]},
        ],
        "derived": [
            {"name": "reach_score", "formula": "log(followers) * (1 + novelty + hook_quality + uses_thread * 0.25) + posting_days * 0.4"},
            {"name": "burnout_risk", "formula": "sigmoid(-2 + posting_days * 0.45 + hook_quality * 0.5)"},
            {"name": "viral", "formula": "reach_score > 18"},
        ],
        "metrics": [
            {"name": "average_reach_score", "op": "mean", "column": "reach_score"},
            {"name": "viral_rate", "op": "rate", "column": "viral"},
            {"name": "average_burnout_risk", "op": "mean", "column": "burnout_risk"},
            {"name": "novelty_reach_correlation", "op": "corr", "x": "novelty", "y": "reach_score"},
        ],
        "charts": [
            {"kind": "histogram", "title": "Reach Score Distribution", "x": "reach_score"},
            {"kind": "scatter", "title": "Novelty vs Reach", "x": "novelty", "y": "reach_score"},
            {"kind": "bar", "title": "Niche Mix", "x": "niche"},
        ],
    }


def custom_model_schema() -> dict[str, Any]:
    return {
        "name": "custom_model",
        "description": "Schema for custom synthetic model specifications.",
        "variables": {
            "description": "Define the input distributions for your synthetic dataset.",
            "types": sorted(SUPPORTED_VARIABLE_TYPES),
            "example": {
                "name": "followers",
                "type": "lognormal",
                "mean": 8.2,
                "sigma": 1.1,
            },
        },
        "derived": {
            "description": "Add derived fields with safe formulas using existing columns and allowed functions.",
            "functions": sorted(SAFE_FUNCTIONS),
            "example": {
                "name": "reach_score",
                "formula": "log(followers) * (1 + novelty + hook_quality) + posting_days * 0.4",
            },
        },
        "metrics": {
            "description": "Compute named summary metrics from dataset fields.",
            "operations": sorted(SUPPORTED_METRIC_OPS),
            "example": {
                "name": "average_reach_score",
                "op": "mean",
                "column": "reach_score",
            },
        },
        "charts": {
            "description": "Generate charts from data fields.",
            "kinds": sorted(SUPPORTED_CHART_KINDS),
            "example": {
                "kind": "histogram",
                "title": "Reach Score Distribution",
                "x": "reach_score",
            },
        },
    }


class CustomModelRunner:
    def run(self, spec: dict[str, Any], size: int = 1000, seed: int | None = None) -> CustomRun:
        normalized = self._normalize_spec(spec)
        dataset = self._generate_dataset(normalized, size, seed)
        self._apply_derived_fields(normalized, dataset)
        metrics = self._compute_metrics(normalized, dataset)
        charts = self._build_charts(normalized)
        insight = self._insight(normalized, metrics, size)

        result = ExperimentResult(
            name=normalized["name"],
            title=normalized["title"],
            dataset=dataset,
            metrics=metrics,
            insight=insight,
            charts=charts,
        )
        return CustomRun(result=result, design=self._design(normalized))

    def _normalize_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(spec, dict):
            raise ValueError("Custom model spec must be a JSON object.")
        normalized = {
            "name": _slug(spec.get("name", "custom_model")),
            "title": str(spec.get("title", "Custom Model Experiment"))[:120],
            "hypothesis": str(spec.get("hypothesis", "The custom model will reveal a measurable synthetic pattern."))[:500],
            "variables": spec.get("variables", []),
            "derived": spec.get("derived", []),
            "metrics": spec.get("metrics", []),
            "charts": spec.get("charts", []),
        }
        if not normalized["variables"]:
            raise ValueError("Custom model needs at least one variable.")
        if len(normalized["variables"]) > 24:
            raise ValueError("Custom model can define up to 24 variables.")
        if len(normalized["derived"]) > 16:
            raise ValueError("Custom model can define up to 16 derived fields.")
        if len(normalized["metrics"]) > 16:
            raise ValueError("Custom model can define up to 16 metrics.")
        if len(normalized["charts"]) > 8:
            raise ValueError("Custom model can define up to 8 charts.")
        self._validate_normalized_spec(normalized)
        return normalized

    def validate(self, spec: dict[str, Any]) -> bool:
        self._normalize_spec(spec)
        return True

    def _generate_dataset(self, spec: dict[str, Any], size: int, seed: int | None) -> pd.DataFrame:
        if size < 10 or size > 10000:
            raise ValueError("Custom model size must be between 10 and 10,000 rows.")

        variable_specs_json = json.dumps(spec["variables"], sort_keys=True)
        return _build_variable_dataset(variable_specs_json, size, seed or 0)

    def _apply_derived_fields(self, spec: dict[str, Any], dataset: pd.DataFrame) -> None:
        for field in spec["derived"]:
            name = _slug(field.get("name", "derived"))
            formula = str(field.get("formula", "")).strip()
            if not formula:
                raise ValueError(f"Derived field '{name}' needs a formula.")
            compiled = field.get("compiled_formula")
            dataset[name] = _safe_eval(formula, dataset, compiled)

    def _compute_metrics(self, spec: dict[str, Any], dataset: pd.DataFrame) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for metric in spec["metrics"]:
            name = _slug(metric.get("name", metric.get("op", "metric")))
            operation = str(metric.get("op", "mean")).lower()
            column = metric.get("column")

            if operation == "corr":
                x = str(metric.get("x"))
                y = str(metric.get("y"))
                _require_columns(dataset, [x, y])
                metrics[name] = correlation(dataset[x].astype(float).to_numpy(), dataset[y].astype(float).to_numpy())
                continue

            if operation == "count":
                metrics[name] = int(len(dataset))
                continue

            if operation == "rate" and metric.get("when"):
                values = _safe_eval(str(metric["when"]), dataset)
                metrics[name] = float(np.mean(values))
                continue

            if column is None:
                raise ValueError(f"Metric '{name}' needs a column.")
            column = str(column)
            _require_columns(dataset, [column])
            values = dataset[column]

            if operation == "mean":
                metrics[name] = float(values.astype(float).mean())
            elif operation == "median":
                metrics[name] = float(values.astype(float).median())
            elif operation == "std":
                metrics[name] = float(values.astype(float).std())
            elif operation == "min":
                metrics[name] = float(values.astype(float).min())
            elif operation == "max":
                metrics[name] = float(values.astype(float).max())
            elif operation == "rate":
                metrics[name] = float(values.astype(bool).mean())
            elif operation == "unique":
                metrics[name] = int(values.nunique())
            else:
                raise ValueError(f"Unsupported metric operation: {operation}")

        if not metrics:
            numeric = dataset.select_dtypes(include=["number", "bool"])
            for column in numeric.columns[:4]:
                metrics[f"average_{column}"] = float(numeric[column].astype(float).mean())
        return metrics

    def _build_charts(self, spec: dict[str, Any]) -> list[ChartSpec]:
        charts: list[ChartSpec] = []
        for chart in spec["charts"]:
            kind = str(chart.get("kind", "histogram")).lower()
            title = str(chart.get("title", f"{kind.title()} Chart"))[:120]
            x = chart.get("x")
            y = chart.get("y")
            charts.append(
                ChartSpec(
                    kind=kind,
                    title=title,
                    filename=f"{_slug(title)}.png",
                    x=str(x) if x else None,
                    y=str(y) if y else None,
                )
            )
        return charts

    def _insight(self, spec: dict[str, Any], metrics: dict[str, Any], size: int) -> str:
        metric_text = ", ".join(f"{key.replace('_', ' ')}={_pretty(value)}" for key, value in list(metrics.items())[:3])
        return f"Ran {size:,} synthetic cases for '{spec['title']}'. The model tested: {spec['hypothesis']} Key outputs: {metric_text}."

    def _design(self, spec: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": spec["name"],
            "title": spec["title"],
            "hypothesis": spec["hypothesis"],
            "synthetic_variables": [_slug(variable.get("name", "field")) for variable in spec["variables"]]
            + [_slug(field.get("name", "derived")) for field in spec["derived"]],
            "metric_plan": [metric.get("name", metric.get("op", "metric")) for metric in spec["metrics"]],
            "visual_plan": [chart.get("title", chart.get("kind", "chart")) for chart in spec["charts"]],
            "run_steps": [
                "parse custom JSON model spec",
                "sample configured distributions",
                "evaluate safe derived formulas",
                "compute custom metrics",
                "return chart-ready result payload",
            ],
        }

    def _validate_normalized_spec(self, spec: dict[str, Any]) -> None:
        variable_names: list[str] = []
        for variable in spec["variables"]:
            name = _slug(variable.get("name", "field"))
            if not name:
                raise ValueError("Every variable requires a valid name.")
            if name in variable_names:
                raise ValueError(f"Duplicate variable name: {name}")
            variable_names.append(name)
            kind = str(variable.get("type", "")).lower()
            if kind not in {"normal", "uniform", "lognormal", "beta", "poisson", "bernoulli", "categorical"}:
                raise ValueError(f"Unsupported variable type: {kind}")
            if kind == "categorical":
                choices = variable.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise ValueError("Categorical variables need a non-empty choices list.")
            if kind == "bernoulli":
                if not 0 <= _float(variable, "p", 0.5) <= 1:
                    raise ValueError("Beroulli probability 'p' must be between 0 and 1.")

        derived_names: list[str] = []
        for derived in spec["derived"]:
            name = _slug(derived.get("name", "derived"))
            if not name:
                raise ValueError("Every derived field requires a valid name.")
            if name in variable_names or name in derived_names:
                raise ValueError(f"Duplicate derived name: {name}")
            derived_names.append(name)
            formula = str(derived.get("formula", "")).strip()
            if not formula:
                raise ValueError(f"Derived field '{name}' needs a formula.")
            compiled = _compile_formula(formula, tuple(variable_names + derived_names))
            derived["compiled_formula"] = compiled

        field_names = set(variable_names + derived_names)
        valid_metric_ops = {"mean", "median", "std", "min", "max", "rate", "count", "unique", "corr"}
        for metric in spec["metrics"]:
            op = str(metric.get("op", "mean")).lower()
            if op not in valid_metric_ops:
                raise ValueError(f"Unsupported metric operation: {op}")
            if op == "corr":
                x = str(metric.get("x", "")).strip()
                y = str(metric.get("y", "")).strip()
                if not x or not y:
                    raise ValueError("Correlation metrics need both x and y columns.")
                if x not in field_names or y not in field_names:
                    raise ValueError(f"Unknown correlation column(s): {x}, {y}")
            elif op == "count":
                continue
            elif op == "rate" and metric.get("when"):
                _validate_formula(str(metric["when"]), variable_names + derived_names)
            else:
                column = metric.get("column")
                if not column:
                    raise ValueError(f"Metric '{metric.get('name', op)}' needs a column.")
                if str(column) not in field_names:
                    raise ValueError(f"Unknown metric column: {column}")

        for chart in spec["charts"]:
            kind = str(chart.get("kind", "histogram")).lower()
            if kind not in {"histogram", "scatter", "bar"}:
                raise ValueError(f"Unsupported chart type: {kind}")
            x = str(chart.get("x", "")).strip()
            y = str(chart.get("y", "")).strip()
            if kind == "scatter":
                if not x or not y:
                    raise ValueError("Scatter charts need both x and y values.")
                if x not in field_names or y not in field_names:
                    raise ValueError(f"Unknown scatter axis columns: {x}, {y}")
            elif kind in {"histogram", "bar"}:
                if not x:
                    raise ValueError(f"{kind.title()} charts need an x column.")
                if x not in field_names:
                    raise ValueError(f"Unknown chart column: {x}")


def _validate_formula(formula: str, allowed_columns: list[str]) -> None:
    _compile_formula(formula, tuple(allowed_columns))


def _safe_eval(formula: str, dataset: pd.DataFrame, compiled: Any | None = None) -> Any:
    if compiled is None:
        compiled = _compile_formula(formula, tuple(dataset.columns))
    local_scope = {column: dataset[column].to_numpy() for column in dataset.columns}
    local_scope.update(SAFE_FUNCTIONS)
    return eval(compiled, {"__builtins__": {}}, local_scope)


def _require_columns(dataset: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in dataset.columns]
    if missing:
        raise ValueError(f"Unknown column(s): {', '.join(missing)}")


def _float(mapping: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return default


def _slug(value: Any) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value).strip().lower()).strip("_")
    return slug[:64] or "field"


def _pretty(value: Any) -> str:
    if isinstance(value, float):
        if 0 <= value <= 1:
            return f"{value * 100:.1f}%"
        return f"{value:,.2f}"
    return str(value)
