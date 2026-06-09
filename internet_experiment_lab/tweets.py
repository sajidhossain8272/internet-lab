from __future__ import annotations

from itertools import count

from internet_experiment_lab.core import ExperimentResult


class TweetGenerator:
    _counter = count(1)

    def generate(self, result: ExperimentResult, experiment_number: int | None = None) -> str:
        number = experiment_number if experiment_number is not None else next(self._counter)
        standout_metric = self._standout_metric(result)
        return (
            f"Internet Experiment #{number}: I simulated {len(result.dataset):,} synthetic cases for "
            f"{result.title.lower()}. {standout_metric} {result.insight}"
        )[:280]

    def _standout_metric(self, result: ExperimentResult) -> str:
        priority = [
            "viral_rate",
            "offer_rate",
            "simulated_crack_rate",
            "procrastination_rate",
            "high_debt_burden_rate",
        ]
        for key in priority:
            if key in result.metrics:
                return f"{key.replace('_', ' ').title()}: {result.metrics[key] * 100:.1f}%."

        first_key, first_value = next(iter(result.metrics.items()))
        if isinstance(first_value, float):
            return f"{first_key.replace('_', ' ').title()}: {first_value:.2f}."
        return f"{first_key.replace('_', ' ').title()}: {first_value}."
