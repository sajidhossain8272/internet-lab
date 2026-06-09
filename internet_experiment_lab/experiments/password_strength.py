from __future__ import annotations

import numpy as np
import pandas as pd

from internet_experiment_lab.core import BaseExperiment, ChartSpec, ExperimentResult
from internet_experiment_lab.math_utils import correlation, sigmoid


class PasswordStrengthExperiment(BaseExperiment):
    name = "passwords"
    title = "Password Strength Simulation"
    description = "Models entropy, composition, common-pattern risk, and crack failure rates."

    def run(self, size: int = 1000, seed: int | None = None) -> ExperimentResult:
        rng = np.random.default_rng(seed)
        length = rng.integers(6, 25, size=size)
        has_upper = rng.random(size) < 0.62
        has_number = rng.random(size) < 0.74
        has_symbol = rng.random(size) < 0.38
        uses_common_word = rng.random(size) < 0.28
        reused_password = rng.random(size) < 0.22

        alphabet_size = 26 + has_upper * 26 + has_number * 10 + has_symbol * 28
        entropy_bits = length * np.log2(alphabet_size) - uses_common_word * 18 - reused_password * 10
        entropy_bits = entropy_bits.clip(4)
        crack_probability = sigmoid(3.6 - entropy_bits / 13 + uses_common_word * 1.4 + reused_password * 1.1)
        cracked = rng.random(size) < crack_probability
        strength = np.select(
            [entropy_bits < 45, entropy_bits < 70, entropy_bits < 95],
            ["weak", "medium", "strong"],
            default="elite",
        )

        dataset = pd.DataFrame(
            {
                "length": length,
                "has_upper": has_upper,
                "has_number": has_number,
                "has_symbol": has_symbol,
                "uses_common_word": uses_common_word,
                "reused_password": reused_password,
                "entropy_bits": entropy_bits.round(2),
                "crack_probability": crack_probability.round(4),
                "cracked": cracked,
                "strength": strength,
            }
        )

        metrics = {
            "average_entropy_bits": float(dataset["entropy_bits"].mean()),
            "median_entropy_bits": float(dataset["entropy_bits"].median()),
            "weak_password_rate": float((dataset["strength"] == "weak").mean()),
            "simulated_crack_rate": float(dataset["cracked"].mean()),
            "common_word_rate": float(dataset["uses_common_word"].mean()),
            "length_entropy_correlation": correlation(dataset["length"].to_numpy(), dataset["entropy_bits"].to_numpy()),
        }

        insight = (
            f"The password lab cracked {metrics['simulated_crack_rate'] * 100:.1f}% of synthetic passwords. "
            f"Average entropy landed at {metrics['average_entropy_bits']:.1f} bits, and length had r="
            f"{metrics['length_entropy_correlation']:.2f} with entropy."
        )

        return ExperimentResult(
            name=self.name,
            title=self.title,
            dataset=dataset,
            metrics=metrics,
            insight=insight,
            charts=[
                ChartSpec("histogram", "Password Entropy Distribution", "entropy_distribution.png", x="entropy_bits", bins=30),
                ChartSpec("scatter", "Length vs Entropy", "length_vs_entropy.png", x="length", y="entropy_bits"),
                ChartSpec("bar", "Password Strength Buckets", "strength.png", x="strength"),
            ],
        )
