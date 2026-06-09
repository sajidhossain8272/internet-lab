from __future__ import annotations

import numpy as np
import pandas as pd

from internet_experiment_lab.core import BaseExperiment, ChartSpec, ExperimentResult
from internet_experiment_lab.math_utils import correlation, gini


class EconomyExperiment(BaseExperiment):
    name = "economy"
    title = "Synthetic Economy Simulation"
    description = "Models income, spending, savings, debt burden, and inequality."

    def run(self, size: int = 1000, seed: int | None = None) -> ExperimentResult:
        rng = np.random.default_rng(seed)

        income = rng.lognormal(mean=10.7, sigma=0.75, size=size)
        savings_rate = np.clip(rng.beta(2.2, 6.5, size=size) + income / income.max() * 0.08, 0, 0.75)
        rent_pressure = rng.normal(0.33, 0.11, size=size).clip(0.08, 0.72)
        discretionary_noise = rng.normal(0, 4200, size=size)
        spending = (income * (1 - savings_rate) * (0.78 + rent_pressure / 2) + discretionary_noise).clip(2000)
        debt = rng.gamma(shape=2.1, scale=6500, size=size) * (1.25 - savings_rate)
        debt_burden = debt / income
        segment = np.select(
            [income < 35000, income < 90000],
            ["stretched", "middle"],
            default="high_income",
        )

        dataset = pd.DataFrame(
            {
                "income": income.round(2),
                "spending": spending.round(2),
                "savings_rate": savings_rate.round(4),
                "debt": debt.round(2),
                "debt_burden": debt_burden.round(4),
                "segment": segment,
            }
        )

        metrics = {
            "average_income": float(dataset["income"].mean()),
            "median_income": float(dataset["income"].median()),
            "average_savings_rate": float(dataset["savings_rate"].mean()),
            "income_gini": gini(dataset["income"].to_numpy()),
            "high_debt_burden_rate": float((dataset["debt_burden"] > 0.45).mean()),
            "income_spending_correlation": correlation(dataset["income"].to_numpy(), dataset["spending"].to_numpy()),
        }

        insight = (
            f"The synthetic economy produced a median income of ${metrics['median_income']:,.0f}, "
            f"with {metrics['high_debt_burden_rate'] * 100:.1f}% of people carrying debt above 45% of income. "
            f"Income and spending moved together at r={metrics['income_spending_correlation']:.2f}."
        )

        return ExperimentResult(
            name=self.name,
            title=self.title,
            dataset=dataset,
            metrics=metrics,
            insight=insight,
            charts=[
                ChartSpec("histogram", "Income Distribution", "income_distribution.png", x="income", bins=35),
                ChartSpec("scatter", "Income vs Spending", "income_vs_spending.png", x="income", y="spending"),
                ChartSpec("bar", "Economic Segments", "segments.png", x="segment"),
            ],
        )
