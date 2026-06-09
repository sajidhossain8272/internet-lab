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


@dataclass(frozen=True)
class ExperimentDesign:
    name: str
    title: str
    hypothesis: str
    synthetic_variables: list[str]
    metric_plan: list[str]
    visual_plan: list[str]
    run_steps: list[str]


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
        normalized_name = self._normalize_name(normalized_name)

        if normalized_name not in self._experiments:
            available = ", ".join(sorted(self._experiments))
            raise ValueError(f"Unknown experiment '{name}'. Available: {available}")

        if size < 10:
            raise ValueError("Experiment size must be at least 10 rows.")

        return self._experiments[normalized_name]().run(size=size, seed=seed)

    def design(self, name: str) -> ExperimentDesign:
        normalized_name = self._normalize_name(name.lower().strip())
        if normalized_name not in self._experiments:
            available = ", ".join(sorted(self._experiments))
            raise ValueError(f"Unknown experiment '{name}'. Available: {available}")

        experiment = self._experiments[normalized_name]
        design_library = {
            "economy": ExperimentDesign(
                name=experiment.name,
                title=experiment.title,
                hypothesis="Debt stress and savings behavior split a synthetic economy into visible pressure groups.",
                synthetic_variables=["income", "spending", "savings_rate", "debt", "debt_burden", "segment"],
                metric_plan=["average and median income", "savings rate", "income Gini", "high debt burden rate", "income-spending correlation"],
                visual_plan=["income histogram", "income versus spending scatter", "economic segment bar chart"],
                run_steps=["sample log-normal incomes", "model spending and debt pressure", "segment households", "compute inequality and correlation metrics"],
            ),
            "attention": ExperimentDesign(
                name=experiment.name,
                title=experiment.title,
                hypothesis="Novelty, outrage, and topic choice create a small number of outsized attention spikes.",
                synthetic_variables=["topic", "followers", "novelty", "outrage", "clarity", "attention_score", "shares", "viral"],
                metric_plan=["viral rate", "average attention", "average shares", "novelty-attention correlation", "top viral topic"],
                visual_plan=["attention histogram", "novelty versus attention scatter", "topic bar chart"],
                run_steps=["sample post topics and account sizes", "score novelty and emotional charge", "mark viral outliers", "compare topic-level lift"],
            ),
            "jobs": ExperimentDesign(
                name=experiment.name,
                title=experiment.title,
                hypothesis="Skill matters, but network strength changes who gets offers at the margin.",
                synthetic_variables=["skill_score", "experience_years", "network_strength", "education", "offer_probability", "received_offer", "salary"],
                metric_plan=["offer rate", "salary when offered", "skill gap", "network-offer correlation"],
                visual_plan=["salary histogram", "network versus offer probability scatter", "education bar chart"],
                run_steps=["sample applicant profiles", "estimate offer probability", "simulate hiring outcomes", "summarize salary and rejection gaps"],
            ),
            "passwords": ExperimentDesign(
                name=experiment.name,
                title=experiment.title,
                hypothesis="Length and character variety raise entropy, but common patterns erase a surprising amount of safety.",
                synthetic_variables=["length", "has_upper", "has_number", "has_symbol", "uses_common_word", "entropy_bits", "cracked", "strength"],
                metric_plan=["average entropy", "weak password rate", "crack rate", "length-entropy correlation"],
                visual_plan=["entropy histogram", "length versus entropy scatter", "strength bucket bar chart"],
                run_steps=["sample password composition", "estimate entropy", "penalize reused/common patterns", "simulate cracking probability"],
            ),
            "behavior": ExperimentDesign(
                name=experiment.name,
                title=experiment.title,
                hypothesis="Sleep, scrolling, and task completion create measurable mood and focus tradeoffs.",
                synthetic_variables=["sleep_hours", "caffeine_cups", "social_scroll_minutes", "planned_tasks", "completed_tasks", "focus_score", "mood_score"],
                metric_plan=["average sleep", "focus score", "procrastination rate", "impulse purchase rate", "sleep-mood and scroll-focus correlations"],
                visual_plan=["focus histogram", "scrolling versus focus scatter", "procrastination bar chart"],
                run_steps=["sample synthetic daily habits", "score focus and mood", "simulate procrastination", "measure behavioral correlations"],
            ),
        }
        return design_library[normalized_name]

    def list_experiments(self) -> list[ExperimentInfo]:
        return [
            ExperimentInfo(
                name=experiment.name,
                title=experiment.title,
                description=experiment.description,
            )
            for experiment in sorted(self._experiments.values(), key=lambda item: item.name)
        ]

    def _normalize_name(self, name: str) -> str:
        aliases = {
            "social": "attention",
            "social_media": "attention",
            "job": "jobs",
            "job_market": "jobs",
            "password": "passwords",
            "human": "behavior",
            "random_human": "behavior",
        }
        return aliases.get(name, name)
