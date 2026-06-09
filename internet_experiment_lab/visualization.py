from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from internet_experiment_lab.core import ChartSpec, ExperimentResult


class ChartRenderer:
    def __init__(self) -> None:
        plt.style.use("grayscale")

    def render(self, result: ExperimentResult, output_dir: str | Path) -> list[Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        chart_paths: list[Path] = []

        for spec in result.charts:
            path = output_path / spec.filename
            self._render_chart(result.dataset, spec, path)
            chart_paths.append(path)

        return chart_paths

    def _render_chart(self, dataset: pd.DataFrame, spec: ChartSpec, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        if spec.kind == "histogram":
            self._histogram(dataset, spec, ax)
        elif spec.kind == "scatter":
            self._scatter(dataset, spec, ax)
        elif spec.kind == "bar":
            self._bar(dataset, spec, ax)
        else:
            raise ValueError(f"Unsupported chart kind: {spec.kind}")

        ax.set_title(spec.title, fontsize=14, fontweight="bold")
        ax.grid(True, color="#d9d9d9", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def _histogram(self, dataset: pd.DataFrame, spec: ChartSpec, ax: plt.Axes) -> None:
        if spec.x is None:
            raise ValueError("Histogram charts require x.")
        values = dataset[spec.x]
        if spec.x == "salary":
            values = values[values > 0]
        ax.hist(values, bins=spec.bins, color="black", edgecolor="white", alpha=0.85)
        ax.set_xlabel(self._label(spec.x))
        ax.set_ylabel("Count")

    def _scatter(self, dataset: pd.DataFrame, spec: ChartSpec, ax: plt.Axes) -> None:
        if spec.x is None or spec.y is None:
            raise ValueError("Scatter charts require x and y.")
        ax.scatter(dataset[spec.x], dataset[spec.y], s=18, color="black", alpha=0.45, linewidths=0)
        ax.set_xlabel(self._label(spec.x))
        ax.set_ylabel(self._label(spec.y))

    def _bar(self, dataset: pd.DataFrame, spec: ChartSpec, ax: plt.Axes) -> None:
        if spec.x is None:
            raise ValueError("Bar charts require x.")
        counts = dataset[spec.x].astype(str).value_counts().sort_index()
        ax.bar(counts.index, counts.values, color="black", alpha=0.85)
        ax.set_xlabel(self._label(spec.x))
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=25)

    def _label(self, value: str) -> str:
        return value.replace("_", " ").title()
