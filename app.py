from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from internet_experiment_lab.core import ExperimentEngine
from internet_experiment_lab.custom_model import CustomModelRunner, custom_model_schema, default_custom_spec
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

    custom_parser = subparsers.add_parser("run-custom", help="Run a custom JSON model spec")
    custom_parser.add_argument("spec", help="Path to a JSON model spec.")
    custom_parser.add_argument("--size", type=int, default=1000, help="Synthetic rows to generate.")
    custom_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    custom_parser.add_argument(
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


def run_custom_experiment(args: argparse.Namespace) -> int:
    from internet_experiment_lab.visualization import ChartRenderer

    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    custom = CustomModelRunner().run(spec, size=args.size, seed=args.seed)
    result = custom.result

    output_dir = Path(args.output_dir)
    data_dir = output_dir / "datasets"
    chart_dir = output_dir / "charts" / result.name
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / f"{result.name}.csv"
    result.dataset.to_csv(dataset_path, index=False)
    chart_paths = ChartRenderer().render(result, chart_dir)
    tweet = TweetGenerator().generate(result)

    print(f"\nInternet Experiment Lab: {result.title}")
    print("=" * (25 + len(result.title)))
    print(result.insight)
    print("\nCustom design")
    for step in custom.design["run_steps"]:
        print(f"- {step}")
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

        if path.startswith("api/design/"):
            experiment_name = path.split("/", 2)[2]
            design = ExperimentEngine().design(experiment_name)
            return _json_response(start_response, {"design": design.__dict__})

        if path.startswith("api/run/"):
            experiment_name = path.split("/", 2)[2]
            result = _run_web_experiment(experiment_name, query)
            design = ExperimentEngine().design(experiment_name)
            return _json_response(start_response, _web_result_payload(result, design.__dict__))

        if path == "api/custom/template":
            return _json_response(start_response, {"spec": default_custom_spec()})

        if path == "api/custom/schema":
            return _json_response(start_response, {"schema": custom_model_schema()})

        if path == "api/custom/validate":
            payload = _read_json_body(environ)
            custom = CustomModelRunner()
            custom.validate(payload.get("spec", {}))
            return _json_response(start_response, {"valid": True})

        if path == "api/custom/run":
            payload = _read_json_body(environ)
            size = int(payload.get("size", 1000))
            seed = int(payload.get("seed", 42))
            custom = CustomModelRunner().run(payload.get("spec", {}), size=size, seed=seed)
            return _json_response(start_response, _web_result_payload(custom.result, custom.design))

        if path.startswith("run/"):
            experiment_name = path.split("/", 1)[1]
            result = _run_web_experiment(experiment_name, query)
            return _html_response(start_response, _result_page(result))

        if path in {"", "index"}:
            return _html_response(start_response, _home_page())

        return _html_response(start_response, _not_found_page(path), status="404 Not Found")
    except Exception as exc:
        if path.startswith("api/"):
            status = "400 Bad Request" if isinstance(exc, (ValueError, TypeError)) else "500 Internal Server Error"
            return _json_response(start_response, {"error": str(exc)}, status=status)
        return _html_response(start_response, _error_page(exc), status="500 Internal Server Error")


def _run_web_experiment(experiment_name: str, query: dict[str, list[str]]) -> Any:
    size = _int_query(query, "size", 1000)
    seed = _int_query(query, "seed", 42)
    return ExperimentEngine().run(experiment_name, size=size, seed=seed)


def _read_json_body(environ: dict[str, Any]) -> dict[str, Any]:
    length = int(environ.get("CONTENT_LENGTH") or 0)
    body = environ["wsgi.input"].read(length) if length else b"{}"
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object.")
    return payload


def _web_result_payload(result: Any, design: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": result.name,
        "title": result.title,
        "design": design,
        "metrics": result.metrics,
        "insight": result.insight,
        "tweet": TweetGenerator().generate(result),
        "preview": result.dataset.head(12).to_dict(orient="records"),
        "charts": _chart_payloads(result),
    }


def _chart_payloads(result: Any) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    for spec in result.charts:
        payload: dict[str, Any] = {
            "kind": spec.kind,
            "title": spec.title,
            "x": spec.x,
            "y": spec.y,
        }
        if spec.kind == "histogram" and spec.x:
            values = result.dataset[spec.x]
            if spec.x == "salary":
                values = values[values > 0]
            payload["values"] = values.astype(float).round(4).tolist()
        elif spec.kind == "scatter" and spec.x and spec.y:
            sample = result.dataset[[spec.x, spec.y]].head(220)
            payload["points"] = sample.astype(float).round(4).to_dict(orient="records")
        elif spec.kind == "bar" and spec.x:
            counts = result.dataset[spec.x].astype(str).value_counts().sort_index()
            payload["labels"] = counts.index.tolist()
            payload["counts"] = counts.astype(int).tolist()
        charts.append(payload)
    return charts


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
    options = "\n".join(
        f"<option value=\"{escape(experiment.name)}\">{escape(experiment.title)}</option>"
        for experiment in experiments
    )
    experiment_payload = json.dumps([experiment.__dict__ for experiment in experiments])
    custom_template = json.dumps(default_custom_spec(), indent=2)
    return _page(
        "Internet Experiment Lab",
        f"""
        <section class="hero">
          <p class="eyebrow">Synthetic Data Simulation Engine</p>
          <h1>Internet Experiment Lab</h1>
          <p>Design models, run synthetic experiments, watch the lab console, and generate Twitter-ready result pages.</p>
        </section>
        <section class="workbench">
          <div class="tabs" role="tablist" aria-label="Experiment modes">
            <button class="tab active" id="preset-tab" type="button" role="tab" aria-selected="true" aria-controls="preset-panel" data-tab="preset">Preset Runner</button>
            <button class="tab" id="custom-tab" type="button" role="tab" aria-selected="false" aria-controls="custom-panel" data-tab="custom">Custom Model</button>
          </div>
          <section id="preset-panel" class="tab-panel active" role="tabpanel" aria-labelledby="preset-tab">
            <section class="lab-shell">
              <form id="lab-form" class="control-panel">
                <h2>Preset Runner</h2>
                <p class="panel-copy">Run one of the built-in model templates, then inspect the generated synthetic dataset, metrics, charts, and caption.</p>
                <label>Experiment<select id="experiment">{options}</select></label>
                <label>Rows<input id="size" type="number" min="10" max="5000" value="1000"></label>
                <label>Seed<input id="seed" type="number" value="42"></label>
                <button type="submit">Run Experiment</button>
              </form>
              <section class="terminal-panel">
                <div class="terminal-title">internet-lab cli</div>
                <pre id="terminal">$ python app.py run economy
waiting for experiment...</pre>
              </section>
            </section>
          </section>
          <section id="custom-panel" class="tab-panel" role="tabpanel" aria-labelledby="custom-tab" hidden>
            <section class="designer-shell">
              <div class="designer-intro">
                <p class="eyebrow">Custom Model Designer</p>
                <h2>Build any synthetic experiment from a model spec</h2>
                <p>Define distributions, derived formulas, metrics, and charts. The engine runs the spec without arbitrary Python execution.</p>
              </div>
              <section class="editor-help" aria-label="JSON editor instructions">
                <h3>How to use the JSON editor</h3>
                <ol>
                  <li>Start from the starter JSON template and edit the experiment metadata: <strong>name</strong>, <strong>title</strong>, and <strong>hypothesis</strong>.</li>
                  <li>Define synthetic columns in <strong>variables</strong> using distributions like normal, beta, poisson, bernoulli, lognormal, uniform, or categorical.</li>
                  <li>Add <strong>derived</strong> formulas to compute new fields from existing ones, for example <code>sigmoid(-2 + posting_days * 0.45)</code>.</li>
                  <li>Add <strong>metrics</strong> to summarize your dataset with mean, median, rate, count, unique, or correlation values.</li>
                  <li>Add <strong>charts</strong> with histogram, scatter, or bar, then click <strong>Validate JSON</strong> before running your spec.</li>
                </ol>
                <div class="validation-toolbar">
                  <button id="custom-validate" type="button">Validate JSON</button>
                  <div id="custom-validation-output" class="validation-output" aria-live="polite"></div>
                </div>
                <p>Tip: the editor validates JSON before run. Use safe formula functions like <code>log</code>, <code>sqrt</code>, <code>sigmoid</code>, <code>clip</code>, <code>where</code>, <code>minimum</code>, and <code>maximum</code>. The output is synthetic data, not real-world observations.</p>
              </section>
              <form id="custom-form" class="designer-form">
                <div class="designer-controls">
                  <label>Rows<input id="custom-size" type="number" min="10" max="10000" value="1000"></label>
                  <label>Seed<input id="custom-seed" type="number" value="99"></label>
                  <button type="submit">Design & Run Custom Model</button>
                </div>
                <label>Model JSON<textarea id="custom-spec" spellcheck="false">{escape(custom_template)}</textarea></label>
              </form>
            </section>
          </section>
        </section>
        <section id="design-panel" class="design-panel" hidden></section>
        <section id="results" class="results" hidden></section>
        <script>window.EXPERIMENTS = {experiment_payload};</script>
        <script>{_lab_script()}</script>
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


def _lab_script() -> str:
    return r"""
const form = document.querySelector("#lab-form");
const terminal = document.querySelector("#terminal");
const results = document.querySelector("#results");
const designPanel = document.querySelector("#design-panel");
const experimentInput = document.querySelector("#experiment");
const sizeInput = document.querySelector("#size");
const seedInput = document.querySelector("#seed");
const customForm = document.querySelector("#custom-form");
const customSpecInput = document.querySelector("#custom-spec");
const customSizeInput = document.querySelector("#custom-size");
const customSeedInput = document.querySelector("#custom-seed");
const customValidateButton = document.querySelector("#custom-validate");
const customValidationOutput = document.querySelector("#custom-validation-output");
const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".tab-panel");

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const fmt = (value) => {
  if (typeof value === "number") {
    if (value >= 0 && value <= 1) return `${(value * 100).toFixed(1)}%`;
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return String(value);
};
const titleize = (text) => text.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
const addLine = (line) => {
  terminal.textContent += `\n${line}`;
  terminal.scrollTop = terminal.scrollHeight;
};
const setLine = (line) => {
  terminal.textContent = line;
  terminal.scrollTop = terminal.scrollHeight;
};

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.tab;
    tabs.forEach((item) => {
      const isActive = item === tab;
      item.classList.toggle("active", isActive);
      item.setAttribute("aria-selected", String(isActive));
    });
    panels.forEach((panel) => {
      const isActive = panel.id === `${target}-panel`;
      panel.classList.toggle("active", isActive);
      panel.hidden = !isActive;
    });
    results.hidden = true;
    designPanel.hidden = true;
    if (target === "preset") {
      setLine(`$ python app.py run ${experimentInput.value}\nwaiting for experiment...`);
    } else {
      setLine("$ python app.py run-custom custom_model\nedit JSON, then run.");
    }
  });
});

experimentInput.addEventListener("change", () => {
  const name = experimentInput.value;
  setLine(`$ python app.py run ${name}\nwaiting for experiment...`);
  results.hidden = true;
  designPanel.hidden = true;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = experimentInput.value;
  const size = Number(sizeInput.value || 1000);
  const seed = Number(seedInput.value || 42);
  results.hidden = true;
  designPanel.hidden = true;

  setLine(`$ python app.py run ${name} --size ${size} --seed ${seed}`);
  addLine("booting Internet Experiment Lab...");
  await wait(300);
  addLine("loading modular experiment engine...");

  let designPayload;
  try {
    const designResponse = await fetch(`/api/design/${name}`);
    designPayload = await designResponse.json();
  } catch (error) {
    addLine(`design failed: ${error.message}`);
    return;
  }

  const design = designPayload.design;
  await wait(300);
  addLine(`designing hypothesis: ${design.hypothesis}`);
  renderDesign(design);
  await wait(350);
  addLine(`synthetic variables: ${design.synthetic_variables.join(", ")}`);
  await wait(350);
  addLine(`metric plan: ${design.metric_plan.join("; ")}`);
  await wait(350);
  addLine("running simulation...");

  const runPromise = fetch(`/api/run/${name}?size=${size}&seed=${seed}`).then((response) => response.json());
  for (const step of design.run_steps) {
    await wait(420);
    addLine(`> ${step}`);
  }
  addLine("waiting for engine result...");

  let payload;
  try {
    payload = await runPromise;
  } catch (error) {
    addLine(`run failed: ${error.message}`);
    return;
  }

  await wait(250);
  addLine("metrics computed.");
  addLine("rendering charts and result page.");
  renderResults(payload);
  addLine("done.");
});

customForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  results.hidden = true;
  designPanel.hidden = true;

  let spec;
  try {
    spec = JSON.parse(customSpecInput.value);
  } catch (error) {
    setLine("$ internet-lab custom-model\nJSON parse failed.");
    addLine(error.message);
    return;
  }

  const size = Number(customSizeInput.value || 1000);
  const seed = Number(customSeedInput.value || 42);
  const modelName = spec.name || "custom_model";

  setLine(`$ python app.py run-custom ${modelName} --size ${size} --seed ${seed}`);
  addLine("reading custom model spec...");
  await wait(300);
  addLine(`hypothesis: ${spec.hypothesis || "custom synthetic experiment"}`);
  await wait(300);
  addLine(`variables requested: ${(spec.variables || []).map((item) => item.name).join(", ") || "none"}`);
  await wait(300);
  addLine(`derived formulas: ${(spec.derived || []).map((item) => item.name).join(", ") || "none"}`);
  await wait(300);
  addLine("submitting model to simulation engine...");

  let payload;
  try {
    const validateResponse = await fetch("/api/custom/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec }),
    });
    const validatePayload = await validateResponse.json();
    if (!validateResponse.ok) {
      throw new Error(validatePayload.error || "Custom model validation failed.");
    }
    addLine("custom model validated.");

    const response = await fetch("/api/custom/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec, size, seed }),
    });
    payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Custom model failed.");
    }
  } catch (error) {
    addLine(`custom run failed: ${error.message}`);
    return;
  }

  renderDesign(payload.design);
  await wait(350);
  addLine("custom distributions sampled.");
  await wait(350);
  addLine("derived formulas evaluated safely.");
  await wait(350);
  addLine("metrics computed.");
  addLine("rendering custom result page.");
  renderResults(payload);
  addLine("done.");
});

customValidateButton?.addEventListener("click", async () => {
  customValidationOutput.textContent = "Validating JSON...";
  let spec;
  try {
    spec = JSON.parse(customSpecInput.value);
  } catch (error) {
    customValidationOutput.textContent = `JSON parse failed: ${error.message}`;
    return;
  }

  try {
    const response = await fetch("/api/custom/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Validation failed.");
    }
    customValidationOutput.textContent = "JSON is valid. Ready to run.";
  } catch (error) {
    customValidationOutput.textContent = `Validation error: ${error.message}`;
  }
});

function renderDesign(design) {
  designPanel.hidden = false;
  designPanel.innerHTML = `
    <h2>Experiment Design</h2>
    <p>${escapeHtml(design.hypothesis)}</p>
    <div class="design-grid">
      ${designBlock("Variables", design.synthetic_variables)}
      ${designBlock("Metrics", design.metric_plan)}
      ${designBlock("Visuals", design.visual_plan)}
    </div>
  `;
}

function designBlock(title, items) {
  return `<div><h3>${title}</h3><ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>`;
}

function renderResults(payload) {
  results.hidden = false;
  const metrics = Object.entries(payload.metrics)
    .map(([key, value]) => `<li><span>${titleize(key)}</span><strong>${fmt(value)}</strong></li>`)
    .join("");
  const charts = payload.charts.map(renderChart).join("");
  const table = renderTable(payload.preview);
  results.innerHTML = `
    <section class="result-head">
      <p class="eyebrow">Generated Result</p>
      <h2>${escapeHtml(payload.title)}</h2>
      <p>${escapeHtml(payload.insight)}</p>
    </section>
    <section class="tweet">${escapeHtml(payload.tweet)}</section>
    <section><h2>Metrics</h2><ul class="metrics">${metrics}</ul></section>
    <section><h2>Charts</h2><div class="chart-grid">${charts}</div></section>
    <section><h2>Dataset Preview</h2>${table}</section>
  `;
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderChart(chart) {
  if (chart.kind === "histogram") return chartFrame(chart.title, histogramSvg(chart.values || []));
  if (chart.kind === "scatter") return chartFrame(chart.title, scatterSvg(chart.points || [], chart.x, chart.y));
  if (chart.kind === "bar") return chartFrame(chart.title, barSvg(chart.labels || [], chart.counts || []));
  return "";
}

function chartFrame(title, svg) {
  return `<figure class="chart"><figcaption>${escapeHtml(title)}</figcaption>${svg}</figure>`;
}

function histogramSvg(values) {
  if (!values.length) return emptySvg();
  const width = 520, height = 260, pad = 34, bins = 18;
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || 1;
  const counts = Array.from({ length: bins }, () => 0);
  values.forEach((value) => counts[Math.min(bins - 1, Math.floor(((value - min) / span) * bins))]++);
  const maxCount = Math.max(...counts) || 1;
  const barWidth = (width - pad * 2) / bins;
  const bars = counts.map((count, index) => {
    const h = (count / maxCount) * (height - pad * 2);
    const x = pad + index * barWidth;
    const y = height - pad - h;
    return `<rect x="${x}" y="${y}" width="${Math.max(1, barWidth - 2)}" height="${h}" />`;
  }).join("");
  return svgWrap(width, height, `${axis(width, height, pad)}${bars}`);
}

function scatterSvg(points, xKey, yKey) {
  if (!points.length) return emptySvg();
  const width = 520, height = 260, pad = 34;
  const xs = points.map((point) => Number(point[xKey]));
  const ys = points.map((point) => Number(point[yKey]));
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = maxX - minX || 1, spanY = maxY - minY || 1;
  const circles = points.map((point) => {
    const x = pad + ((Number(point[xKey]) - minX) / spanX) * (width - pad * 2);
    const y = height - pad - ((Number(point[yKey]) - minY) / spanY) * (height - pad * 2);
    return `<circle cx="${x}" cy="${y}" r="3" />`;
  }).join("");
  return svgWrap(width, height, `${axis(width, height, pad)}${circles}`);
}

function barSvg(labels, counts) {
  if (!labels.length) return emptySvg();
  const width = 520, height = 260, pad = 34;
  const maxCount = Math.max(...counts) || 1;
  const barWidth = (width - pad * 2) / labels.length;
  const bars = labels.map((label, index) => {
    const h = (counts[index] / maxCount) * (height - pad * 2);
    const x = pad + index * barWidth;
    const y = height - pad - h;
    return `<rect x="${x}" y="${y}" width="${Math.max(8, barWidth - 8)}" height="${h}" /><text x="${x + barWidth / 2}" y="${height - 10}" text-anchor="middle">${escapeHtml(String(label)).slice(0, 10)}</text>`;
  }).join("");
  return svgWrap(width, height, `${axis(width, height, pad)}${bars}`);
}

function axis(width, height, pad) {
  return `<line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" /><line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" />`;
}

function svgWrap(width, height, content) {
  return `<svg viewBox="0 0 ${width} ${height}" role="img">${content}</svg>`;
}

function emptySvg() {
  return svgWrap(520, 260, `<text x="260" y="130" text-anchor="middle">No chart data</text>`);
}

function renderTable(rows) {
  if (!rows.length) return "<p>No preview rows returned.</p>";
  const columns = Object.keys(rows[0]);
  const head = columns.map((column) => `<th>${titleize(column)}</th>`).join("");
  const body = rows.map((row) => `<tr>${columns.map((column) => `<td>${escapeHtml(String(row[column]))}</td>`).join("")}</tr>`).join("");
  return `<div class="table-wrap"><table class="preview"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
"""


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
    h3 {{ margin: 0 0 8px; font-size: 15px; text-transform: uppercase; letter-spacing: .06em; }}
    code {{ border: 1px solid #111; padding: 1px 5px; background: #f5f5f2; font-family: Consolas, "SFMono-Regular", monospace; font-size: .92em; }}
    .hero p:not(.eyebrow) {{ max-width: 720px; font-size: 20px; line-height: 1.55; }}
    .workbench {{ border: 2px solid #111; background: #fff; }}
    .tabs {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); border-bottom: 2px solid #111; }}
    .tab {{ width: 100%; min-height: 54px; border: 0; border-right: 2px solid #111; background: #fff; color: #111; }}
    .tab:last-child {{ border-right: 0; }}
    .tab.active {{ background: #111; color: #fff; }}
    .tab-panel {{ padding: 18px; }}
    .lab-shell {{ display: grid; grid-template-columns: minmax(260px, 340px) 1fr; gap: 18px; align-items: stretch; }}
    .control-panel, .terminal-panel, .design-panel, .designer-shell, .results section, .tweet, .chart {{ border: 2px solid #111; background: #fff; }}
    .control-panel {{ padding: 18px; }}
    .control-panel h2 {{ margin-top: 0; }}
    .panel-copy {{ margin: -4px 0 18px; color: #333; line-height: 1.45; }}
    label {{ display: grid; gap: 7px; margin: 0 0 14px; font-weight: 800; }}
    select, input, textarea {{ width: 100%; min-height: 42px; border: 2px solid #111; background: #fff; color: #111; padding: 8px 10px; font: inherit; }}
    textarea {{ min-height: 460px; resize: vertical; font: 13px/1.45 Consolas, "SFMono-Regular", monospace; }}
    button {{ width: 100%; min-height: 46px; border: 2px solid #111; background: #111; color: #fff; font: inherit; font-weight: 900; cursor: pointer; }}
    button:hover {{ background: #333; }}
    .terminal-panel {{ min-height: 280px; display: grid; grid-template-rows: auto 1fr; background: #111; color: #f8f8f2; }}
    .terminal-title {{ padding: 10px 14px; border-bottom: 1px solid #555; font-size: 13px; text-transform: uppercase; letter-spacing: .08em; }}
    pre {{ margin: 0; padding: 16px; overflow: auto; white-space: pre-wrap; font: 14px/1.55 Consolas, "SFMono-Regular", monospace; }}
    .designer-shell {{ padding: 18px; }}
    .designer-intro {{ border-bottom: 2px solid #111; padding-bottom: 14px; margin-bottom: 18px; }}
    .designer-intro h2 {{ margin-top: 0; font-size: 32px; }}
    .designer-intro p:not(.eyebrow) {{ max-width: 760px; line-height: 1.55; }}
    .editor-help {{ border: 2px solid #111; padding: 16px; margin-bottom: 18px; background: #f5f5f2; }}
    .editor-help ol {{ margin: 10px 0 0; padding-left: 22px; line-height: 1.55; }}
    .editor-help li {{ margin-bottom: 8px; }}
    .editor-help p {{ margin: 12px 0 0; line-height: 1.55; }}
    .designer-form {{ display: grid; grid-template-columns: minmax(220px, 280px) 1fr; gap: 18px; align-items: start; }}
    .designer-controls {{ position: sticky; top: 18px; }}
    .design-panel {{ margin-top: 18px; padding: 18px; }}
    .design-panel h2 {{ margin-top: 0; }}
    .design-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
    .design-grid ul {{ margin: 0; padding-left: 18px; line-height: 1.5; }}
    .results {{ margin-top: 28px; }}
    .results section {{ padding: 18px; margin-bottom: 18px; }}
    .result-head h2 {{ margin-top: 0; font-size: 34px; }}
    .tweet {{ margin: 26px 0; padding: 22px; border: 2px solid #111; background: #fff; font-size: 22px; line-height: 1.4; font-weight: 700; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; padding: 0; list-style: none; }}
    .metrics li {{ display: flex; justify-content: space-between; gap: 16px; border-bottom: 1px solid #111; padding: 12px 0; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    .chart {{ margin: 0; padding: 12px; }}
    .chart figcaption {{ margin-bottom: 8px; font-weight: 900; }}
    svg {{ width: 100%; height: auto; display: block; }}
    svg rect, svg circle {{ fill: #111; opacity: .86; }}
    svg line {{ stroke: #111; stroke-width: 2; }}
    svg text {{ fill: #111; font: 11px system-ui, sans-serif; }}
    .table-wrap {{ overflow-x: auto; }}
    .preview {{ width: 100%; border-collapse: collapse; background: #fff; font-size: 14px; overflow-wrap: anywhere; }}
    .preview th, .preview td {{ border: 1px solid #111; padding: 8px; text-align: left; }}
    @media (max-width: 820px) {{ .lab-shell, .designer-form, .design-grid {{ grid-template-columns: 1fr; }} .designer-controls {{ position: static; }} }}
    @media (max-width: 700px) {{ .tabs {{ grid-template-columns: 1fr; }} .tab {{ border-right: 0; border-bottom: 2px solid #111; }} .tab:last-child {{ border-bottom: 0; }} .tab-panel {{ padding: 12px; }} }}
    @media (max-width: 640px) {{ main {{ width: min(100% - 20px, 1080px); }} h1 {{ font-size: 44px; }} .hero p:not(.eyebrow), .tweet {{ font-size: 17px; }} .result-head h2 {{ font-size: 26px; }} }}
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
    if args.command == "run-custom":
        return run_custom_experiment(args)
    if args.command == "list":
        return list_experiments()

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
