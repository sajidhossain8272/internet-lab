from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules
from typing import Any, ClassVar

import pandas as pd


@dataclass(frozen=True)
class ChartSpec:
    kind: str
    title: str
    filename: str
    x: str | None = None
    y: str | None = None
    bins: int = 30
    metric_keys: list[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    name: str
    title: str
    dataset: pd.DataFrame
    metrics: dict[str, Any]
    insight: str
    charts: list[ChartSpec] = field(default_factory=list)


class BaseExperiment:
    name: ClassVar[str]
    title: ClassVar[str]
    description: ClassVar[str] = ""

    def run(self, size: int = 1000, seed: int | None = None) -> ExperimentResult:
        raise NotImplementedError


@dataclass(frozen=True)
class ExperimentInfo:
    name: str
    title: str
    description: str


class ExperimentEngine:
    def __init__(self) -> None:
        self._experiments: dict[str, type[BaseExperiment]] = {}
        self.discover()

    def discover(self) -> None:
        package_name = "internet_experiment_lab.experiments"
        package = import_module(package_name)
        package_path = Path(package.__file__).parent

        for module_info in iter_modules([str(package_path)]):
            if module_info.name.startswith("_"):
                continue
            module = import_module(f"{package_name}.{module_info.name}")
            for value in module.__dict__.values():
                if (
                    isinstance(value, type)
                    and issubclass(value, BaseExperiment)
                    and value is not BaseExperiment
                ):
                    self.register(value)

    def register(self, experiment: type[BaseExperiment]) -> None:
        self._experiments[experiment.name] = experiment

    def run(self, name: str, size: int = 1000, seed: int | None = None) -> ExperimentResult:
        normalized_name = name.lower().strip()
        aliases = {
            "social": "attention",
            "social_media": "attention",
            "job": "jobs",
            "job_market": "jobs",
            "password": "passwords",
            "human": "behavior",
            "random_human": "behavior",
        }
        normalized_name = aliases.get(normalized_name, normalized_name)

        if normalized_name not in self._experiments:
            available = ", ".join(sorted(self._experiments))
            raise ValueError(f"Unknown experiment '{name}'. Available: {available}")

        if size < 10:
            raise ValueError("Experiment size must be at least 10 rows.")

        return self._experiments[normalized_name]().run(size=size, seed=seed)

    def list_experiments(self) -> list[ExperimentInfo]:
        return [
            ExperimentInfo(
                name=experiment.name,
                title=experiment.title,
                description=experiment.description,
            )
            for experiment in sorted(self._experiments.values(), key=lambda item: item.name)
        ]
