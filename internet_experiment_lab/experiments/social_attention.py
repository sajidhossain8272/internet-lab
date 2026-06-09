from __future__ import annotations

import numpy as np
import pandas as pd

from internet_experiment_lab.core import BaseExperiment, ChartSpec, ExperimentResult
from internet_experiment_lab.math_utils import correlation


class SocialAttentionExperiment(BaseExperiment):
    name = "attention"
    title = "Social Media Attention Model"
    description = "Simulates which posts become attention spikes based on topic, novelty, and outrage."

    def run(self, size: int = 1000, seed: int | None = None) -> ExperimentResult:
        rng = np.random.default_rng(seed)
        topics = np.array(["AI", "money", "dating", "fitness", "politics", "productivity"])
        topic = rng.choice(topics, size=size, p=[0.2, 0.18, 0.14, 0.14, 0.18, 0.16])
        followers = rng.lognormal(mean=8.4, sigma=1.15, size=size).astype(int).clip(25)
        novelty = rng.beta(2.5, 3.2, size=size)
        outrage = rng.beta(1.7, 4.0, size=size)
        clarity = rng.beta(3.0, 2.4, size=size)
        topic_boost = pd.Series(topic).map(
            {"AI": 1.18, "money": 1.12, "dating": 1.07, "fitness": 0.94, "politics": 1.22, "productivity": 0.98}
        ).to_numpy()

        attention = (
            np.log1p(followers)
            * (1 + novelty * 1.5 + outrage * 1.35 + clarity * 0.9)
            * topic_boost
            + rng.normal(0, 2.8, size=size)
        ).clip(0)
        shares = rng.poisson(np.maximum(attention * 0.9, 0.1))
        viral = attention > np.quantile(attention, 0.92)

        dataset = pd.DataFrame(
            {
                "topic": topic,
                "followers": followers,
                "novelty": novelty.round(4),
                "outrage": outrage.round(4),
                "clarity": clarity.round(4),
                "attention_score": attention.round(3),
                "shares": shares,
                "viral": viral,
            }
        )

        topic_rates = dataset.groupby("topic")["viral"].mean().sort_values(ascending=False)
        metrics = {
            "average_attention_score": float(dataset["attention_score"].mean()),
            "viral_rate": float(dataset["viral"].mean()),
            "average_shares": float(dataset["shares"].mean()),
            "outrage_attention_correlation": correlation(dataset["outrage"].to_numpy(), dataset["attention_score"].to_numpy()),
            "novelty_attention_correlation": correlation(dataset["novelty"].to_numpy(), dataset["attention_score"].to_numpy()),
            "top_viral_topic": str(topic_rates.index[0]),
        }

        insight = (
            f"Only {metrics['viral_rate'] * 100:.1f}% of posts crossed the viral threshold. "
            f"{metrics['top_viral_topic']} had the strongest viral rate, while novelty correlated with attention "
            f"at r={metrics['novelty_attention_correlation']:.2f}."
        )

        return ExperimentResult(
            name=self.name,
            title=self.title,
            dataset=dataset,
            metrics=metrics,
            insight=insight,
            charts=[
                ChartSpec("histogram", "Attention Score Distribution", "attention_distribution.png", x="attention_score", bins=30),
                ChartSpec("scatter", "Novelty vs Attention", "novelty_vs_attention.png", x="novelty", y="attention_score"),
                ChartSpec("bar", "Posts by Topic", "topics.png", x="topic"),
            ],
        )
