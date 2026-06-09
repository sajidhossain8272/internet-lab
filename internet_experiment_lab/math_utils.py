from __future__ import annotations

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def gini(values: np.ndarray) -> float:
    sorted_values = np.sort(np.asarray(values, dtype=float))
    if len(sorted_values) == 0:
        return 0.0
    if np.sum(sorted_values) == 0:
        return 0.0
    index = np.arange(1, len(sorted_values) + 1)
    return float((2 * np.sum(index * sorted_values)) / (len(sorted_values) * np.sum(sorted_values)) - (len(sorted_values) + 1) / len(sorted_values))
