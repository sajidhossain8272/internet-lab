from __future__ import annotations

import argparse
from pathlib import Path

from internet_experiment_lab.core import ExperimentEngine
from internet_experiment_lab.tweets import TweetGenerator
from internet_experiment_lab.visualization import ChartRenderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Internet Experiment Lab",
        description="Run synthetic viral data experiments from one command.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument(
        "experiment",
        help="Experiment name, such as economy, attention, jobs, passwords, or behavior.",
    )
    run_parser.add_argument("--size", type=int, default=1000, help="Synthetic rows to generate.")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    run_parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for generated datasets and charts.",
    )

    subparsers.add_parser("list", help="List available experiments")
    return parser


def run_experiment(args: argparse.Namespace) -> int:
    engine = ExperimentEngine()
    result = engine.run(args.experiment, size=args.size, seed=args.seed)

    output_dir = Path(args.output_dir)
    data_dir = output_dir / "datasets"
    chart_dir = output_dir / "charts" / result.name
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / f"{result.name}.csv"
    result.dataset.to_csv(dataset_path, index=False)

    renderer = ChartRenderer()
    chart_paths = renderer.render(result, chart_dir)

    tweet = TweetGenerator().generate(result)

    print(f"\nInternet Experiment Lab: {result.title}")
    print("=" * (25 + len(result.title)))
    print(result.insight)
    print("\nMetrics")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"- {key}: {value:,.3f}")
        else:
            print(f"- {key}: {value}")

    print("\nTweet-ready caption")
    print(tweet)

    print("\nSaved outputs")
    print(f"- dataset: {dataset_path}")
    for chart_path in chart_paths:
        print(f"- chart: {chart_path}")

    return 0


def list_experiments() -> int:
    engine = ExperimentEngine()
    print("Available experiments:")
    for experiment in engine.list_experiments():
        print(f"- {experiment.name}: {experiment.title}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return run_experiment(args)
    if args.command == "list":
        return list_experiments()

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
