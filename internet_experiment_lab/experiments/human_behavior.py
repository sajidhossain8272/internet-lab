from __future__ import annotations

import numpy as np
import pandas as pd

from internet_experiment_lab.core import BaseExperiment, ChartSpec, ExperimentResult
from internet_experiment_lab.math_utils import correlation, sigmoid


class HumanBehaviorExperiment(BaseExperiment):
    name = "behavior"
    title = "Random Human Behavior Simulation"
    description = "Simulates daily sleep, focus, scrolling, procrastination, spending impulses, and mood."

    def run(self, size: int = 1000, seed: int | None = None) -> ExperimentResult:
        rng = np.random.default_rng(seed)
        sleep_hours = rng.normal(7.1, 1.25, size=size).clip(3.5, 10.5)
        caffeine_cups = rng.poisson(1.8, size=size).clip(0, 8)
        social_scroll_minutes = rng.gamma(2.2, 38, size=size).clip(0, 360)
        planned_tasks = rng.poisson(5.5, size=size).clip(1, 16)
        focus_score = (72 + sleep_hours * 4.2 - social_scroll_minutes * 0.12 + caffeine_cups * 1.3 + rng.normal(0, 10, size=size)).clip(0, 100)
        procrastinated = rng.random(size) < sigmoid(1.0 - focus_score / 22 + social_scroll_minutes / 95)
        impulse_purchase = rng.random(size) < sigmoid(-2.0 + social_scroll_minutes / 130 + caffeine_cups * 0.12)
        completed_tasks = np.where(
            procrastinated,
            rng.binomial(planned_tasks, 0.46),
            rng.binomial(planned_tasks, 0.78),
        )
        mood_score = (45 + sleep_hours * 5.5 + completed_tasks * 1.4 - procrastinated * 8 - social_scroll_minutes * 0.04 + rng.normal(0, 8, size=size)).clip(0, 100)

        dataset = pd.DataFrame(
            {
                "sleep_hours": sleep_hours.round(2),
                "caffeine_cups": caffeine_cups,
                "social_scroll_minutes": social_scroll_minutes.round(1),
                "planned_tasks": planned_tasks,
                "completed_tasks": completed_tasks,
                "focus_score": focus_score.round(2),
                "procrastinated": procrastinated,
                "impulse_purchase": impulse_purchase,
                "mood_score": mood_score.round(2),
            }
        )

        metrics = {
            "average_sleep_hours": float(dataset["sleep_hours"].mean()),
            "average_focus_score": float(dataset["focus_score"].mean()),
            "procrastination_rate": float(dataset["procrastinated"].mean()),
            "impulse_purchase_rate": float(dataset["impulse_purchase"].mean()),
            "sleep_mood_correlation": correlation(dataset["sleep_hours"].to_numpy(), dataset["mood_score"].to_numpy()),
            "scroll_focus_correlation": correlation(dataset["social_scroll_minutes"].to_numpy(), dataset["focus_score"].to_numpy()),
        }

        insight = (
            f"Across synthetic days, procrastination appeared in {metrics['procrastination_rate'] * 100:.1f}% of cases. "
            f"Sleep correlated with mood at r={metrics['sleep_mood_correlation']:.2f}, while scrolling pulled against "
            f"focus at r={metrics['scroll_focus_correlation']:.2f}."
        )

        return ExperimentResult(
            name=self.name,
            title=self.title,
            dataset=dataset,
            metrics=metrics,
            insight=insight,
            charts=[
                ChartSpec("histogram", "Focus Score Distribution", "focus_distribution.png", x="focus_score", bins=30),
                ChartSpec("scatter", "Scrolling vs Focus", "scrolling_vs_focus.png", x="social_scroll_minutes", y="focus_score"),
                ChartSpec("bar", "Procrastination Outcomes", "procrastination.png", x="procrastinated"),
            ],
        )
