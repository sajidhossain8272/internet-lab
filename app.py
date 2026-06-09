from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from internet_experiment_lab.core import ExperimentEngine
from internet_experiment_lab.tweets import TweetGenerator


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
    from internet_experiment_lab.visualization import ChartRenderer

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


def app(environ: dict[str, Any], start_response: Any) -> list[bytes]:
    """Small WSGI app for Vercel deployment while preserving the CLI."""
    path = environ.get("PATH_INFO", "/").strip("/") or ""
    query = parse_qs(environ.get("QUERY_STRING", ""))

    try:
        if path == "api/experiments":
            engine = ExperimentEngine()
            payload = [
                {
                    "name": experiment.name,
                    "title": experiment.title,
                    "description": experiment.description,
                }
                for experiment in engine.list_experiments()
            ]
            return _json_response(start_response, {"experiments": payload})

        if path.startswith("api/run/"):
            experiment_name = path.split("/", 2)[2]
            result = _run_web_experiment(experiment_name, query)
            return _json_response(
                start_response,
                {
                    "name": result.name,
                    "title": result.title,
                    "metrics": result.metrics,
                    "insight": result.insight,
                    "tweet": TweetGenerator().generate(result),
                    "preview": result.dataset.head(10).to_dict(orient="records"),
                },
            )

        if path.startswith("run/"):
            experiment_name = path.split("/", 1)[1]
            result = _run_web_experiment(experiment_name, query)
            return _html_response(start_response, _result_page(result))

        if path in {"", "index"}:
            return _html_response(start_response, _home_page())

        return _html_response(start_response, _not_found_page(path), status="404 Not Found")
    except Exception as exc:
        return _html_response(start_response, _error_page(exc), status="500 Internal Server Error")


def _run_web_experiment(experiment_name: str, query: dict[str, list[str]]) -> Any:
    size = _int_query(query, "size", 1000)
    seed = _int_query(query, "seed", 42)
    return ExperimentEngine().run(experiment_name, size=size, seed=seed)


def _int_query(query: dict[str, list[str]], key: str, default: int) -> int:
    try:
        return int(query.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


def _html_response(start_response: Any, body: str, status: str = "200 OK") -> list[bytes]:
    payload = body.encode("utf-8")
    start_response(
        status,
        [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(payload))),
        ],
    )
    return [payload]


def _json_response(start_response: Any, payload: dict[str, Any], status: str = "200 OK") -> list[bytes]:
    data = json.dumps(payload, default=str, indent=2).encode("utf-8")
    start_response(
        status,
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(data))),
        ],
    )
    return [data]


def _home_page() -> str:
    engine = ExperimentEngine()
    experiments = engine.list_experiments()
    links = "\n".join(
        f"""
        <a class="experiment" href="/run/{escape(experiment.name)}">
          <strong>{escape(experiment.title)}</strong>
          <span>{escape(experiment.description)}</span>
        </a>
        """
        for experiment in experiments
    )
    return _page(
        "Internet Experiment Lab",
        f"""
        <section class="hero">
          <p class="eyebrow">Synthetic Data Simulation Engine</p>
          <h1>Internet Experiment Lab</h1>
          <p>Run economy, attention, jobs, passwords, and behavior simulations with Twitter-ready summaries.</p>
        </section>
        <section class="grid">{links}</section>
        """,
    )


def _result_page(result: Any) -> str:
    metrics = "\n".join(
        f"<li><span>{escape(key.replace('_', ' ').title())}</span><strong>{_format_metric(value)}</strong></li>"
        for key, value in result.metrics.items()
    )
    preview = result.dataset.head(8).to_html(index=False, classes="preview")
    tweet = escape(TweetGenerator().generate(result))
    return _page(
        result.title,
        f"""
        <nav><a href="/">Back to experiments</a><a href="/api/run/{escape(result.name)}">JSON API</a></nav>
        <section class="hero compact">
          <p class="eyebrow">Internet Experiment</p>
          <h1>{escape(result.title)}</h1>
          <p>{escape(result.insight)}</p>
        </section>
        <section class="tweet">{tweet}</section>
        <section><h2>Metrics</h2><ul class="metrics">{metrics}</ul></section>
        <section><h2>Dataset Preview</h2>{preview}</section>
        """,
    )


def _not_found_page(path: str) -> str:
    return _page("Not Found", f"<section class='hero compact'><h1>Not Found</h1><p>No route for /{escape(path)}</p></section>")


def _error_page(exc: Exception) -> str:
    return _page("Error", f"<section class='hero compact'><h1>Experiment failed</h1><p>{escape(str(exc))}</p></section>")


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #f5f5f2; color: #111; }}
    main {{ width: min(1080px, calc(100% - 32px)); margin: 0 auto; padding: 42px 0 64px; }}
    nav {{ display: flex; gap: 18px; margin-bottom: 28px; }}
    a {{ color: inherit; }}
    .hero {{ border-bottom: 2px solid #111; padding: 44px 0 34px; margin-bottom: 28px; }}
    .hero.compact {{ padding-top: 18px; }}
    .eyebrow {{ margin: 0 0 12px; text-transform: uppercase; letter-spacing: .08em; font-size: 13px; font-weight: 800; }}
    h1 {{ margin: 0; font-size: clamp(42px, 8vw, 84px); line-height: .95; max-width: 900px; }}
    h2 {{ margin: 36px 0 14px; font-size: 22px; }}
    .hero p:not(.eyebrow) {{ max-width: 720px; font-size: 20px; line-height: 1.55; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; }}
    .experiment {{ display: block; min-height: 150px; padding: 18px; border: 2px solid #111; text-decoration: none; background: #fff; }}
    .experiment strong {{ display: block; font-size: 20px; margin-bottom: 12px; }}
    .experiment span {{ color: #333; line-height: 1.45; }}
    .tweet {{ margin: 26px 0; padding: 22px; border: 2px solid #111; background: #fff; font-size: 22px; line-height: 1.4; font-weight: 700; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; padding: 0; list-style: none; }}
    .metrics li {{ display: flex; justify-content: space-between; gap: 16px; border-bottom: 1px solid #111; padding: 12px 0; }}
    .preview {{ width: 100%; border-collapse: collapse; background: #fff; font-size: 14px; overflow-wrap: anywhere; }}
    .preview th, .preview td {{ border: 1px solid #111; padding: 8px; text-align: left; }}
    @media (max-width: 640px) {{ main {{ width: min(100% - 20px, 1080px); }} h1 {{ font-size: 44px; }} .hero p:not(.eyebrow), .tweet {{ font-size: 17px; }} }}
  </style>
</head>
<body>
  <main>{body}</main>
</body>
</html>"""


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        if 0 <= value <= 1:
            return f"{value * 100:.1f}%"
        return f"{value:,.2f}"
    return escape(str(value))


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
