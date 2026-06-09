from __future__ import annotations

import numpy as np
import pandas as pd

from internet_experiment_lab.core import BaseExperiment, ChartSpec, ExperimentResult
from internet_experiment_lab.math_utils import correlation, sigmoid


class JobMarketExperiment(BaseExperiment):
    name = "jobs"
    title = "Job Market Simulation"
    description = "Simulates applicants, offers, salary outcomes, and network effects."

    def run(self, size: int = 1000, seed: int | None = None) -> ExperimentResult:
        rng = np.random.default_rng(seed)
        skill = rng.normal(0, 1, size=size)
        experience_years = rng.poisson(5.2, size=size).clip(0, 35)
        network_strength = rng.beta(2.1, 3.5, size=size)
        education = rng.choice(["bootcamp", "bachelor", "master", "self_taught"], size=size, p=[0.18, 0.42, 0.2, 0.2])
        education_boost = pd.Series(education).map({"bootcamp": 0.0, "bachelor": 0.25, "master": 0.42, "self_taught": -0.05}).to_numpy()

        offer_probability = sigmoid(-1.2 + skill * 0.95 + experience_years * 0.08 + network_strength * 1.6 + education_boost)
        offer = rng.random(size) < offer_probability
        salary = np.where(
            offer,
            52000 + skill * 10500 + experience_years * 2300 + network_strength * 13000 + education_boost * 15000 + rng.normal(0, 8500, size),
            0,
        ).clip(0)

        dataset = pd.DataFrame(
            {
                "skill_score": skill.round(3),
                "experience_years": experience_years,
                "network_strength": network_strength.round(4),
                "education": education,
                "offer_probability": offer_probability.round(4),
                "received_offer": offer,
                "salary": salary.round(2),
            }
        )

        offered = dataset[dataset["received_offer"]]
        not_offered = dataset[~dataset["received_offer"]]
        metrics = {
            "offer_rate": float(dataset["received_offer"].mean()),
            "average_salary_when_offered": float(offered["salary"].mean()) if not offered.empty else 0.0,
            "median_salary_when_offered": float(offered["salary"].median()) if not offered.empty else 0.0,
            "network_offer_correlation": correlation(dataset["network_strength"].to_numpy(), dataset["received_offer"].astype(int).to_numpy()),
            "average_skill_offered": float(offered["skill_score"].mean()) if not offered.empty else 0.0,
            "average_skill_rejected": float(not_offered["skill_score"].mean()) if not not_offered.empty else 0.0,
        }

        insight = (
            f"The simulated offer rate was {metrics['offer_rate'] * 100:.1f}%. "
            f"People with offers averaged {metrics['average_skill_offered']:.2f} skill score versus "
            f"{metrics['average_skill_rejected']:.2f} for rejected applicants, with network strength showing r="
            f"{metrics['network_offer_correlation']:.2f} against offer outcomes."
        )

        return ExperimentResult(
            name=self.name,
            title=self.title,
            dataset=dataset,
            metrics=metrics,
            insight=insight,
            charts=[
                ChartSpec("histogram", "Salary Distribution for Offers", "salary_distribution.png", x="salary", bins=28),
                ChartSpec("scatter", "Network Strength vs Offer Probability", "network_vs_offer_probability.png", x="network_strength", y="offer_probability"),
                ChartSpec("bar", "Applicants by Education", "education.png", x="education"),
            ],
        )
