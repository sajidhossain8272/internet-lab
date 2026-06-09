# Internet Experiment Lab

Internet Experiment Lab is a synthetic data simulation engine for Twitter-ready data experiments. It generates datasets with NumPy and Pandas, computes metrics, writes clean black-and-white Matplotlib charts, and prints a short viral-style caption.

## Run

```bash
python app.py list
python app.py run economy
python app.py run attention --size 2000 --seed 7
python app.py run jobs
python app.py run passwords
python app.py run behavior
python app.py run-custom examples/creator_growth_model.json --size 2000 --seed 11
```

The same `app.py` also exposes a top-level WSGI `app` callable for Vercel. The deployed web routes are:

- `/` for the experiment launcher
- `/run/economy`, `/run/attention`, `/run/jobs`, `/run/passwords`, `/run/behavior`
- `/api/design/<experiment>` for the generated experiment design plan
- `/api/run/<experiment>` for JSON output
- `/api/custom/template` for a starter custom model spec
- `/api/custom/run` for running custom JSON model specs

On the Vercel homepage, the lab now uses two tabs:

- `Preset Runner`: select a built-in experiment, set the row count and seed, and run it.
- `Custom Model`: switch to the JSON editor tab, edit the starter spec, and run your own custom synthetic experiment.

How to use the JSON editor:

1. Open the `Custom Model` tab on the homepage.
2. Edit the starter JSON template, updating `name`, `title`, and `hypothesis`.
3. Define `variables` and optional `derived` fields, then add `metrics` and `charts`.
4. Click `Design & Run Custom Model` to validate and execute the spec.
5. Inspect the generated dataset preview, metrics, and charts in the results panel.

Both modes show a terminal-style progress log, then generate the result page, SVG charts, metrics, tweet caption, and dataset preview dynamically.

Outputs are saved under `outputs/`:

- `outputs/datasets/<experiment>.csv`
- `outputs/charts/<experiment>/*.png`

## Project Structure

```text
app.py
internet_experiment_lab/
  core.py
  math_utils.py
  tweets.py
  visualization.py
  custom_model.py
  experiments/
    economy.py
    social_attention.py
    job_market.py
    password_strength.py
    human_behavior.py
examples/
  creator_growth_model.json
```

## Custom Model Specs

Custom specs let you design new synthetic experiments without writing Python. Supported variable distributions:

- `normal`: `mean`, `sd`
- `uniform`: `min`, `max`
- `lognormal`: `mean`, `sigma`
- `beta`: `a`, `b`
- `poisson`: `lambda`
- `bernoulli`: `p`
- `categorical`: `choices`, optional `probabilities`

Derived fields use safe vector formulas over existing columns:

```json
{
  "name": "reach_score",
  "formula": "log(followers) * (1 + novelty + hook_quality) + posting_days * 0.4"
}
```

Supported formula functions are `abs`, `clip`, `exp`, `log`, `maximum`, `minimum`, `round`, `sigmoid`, `sqrt`, and `where`.

Supported metric operations are `mean`, `median`, `std`, `min`, `max`, `rate`, `count`, `unique`, and `corr`.

Supported chart types are `histogram`, `scatter`, and `bar`.

## Why this project is useful

This repository is a lightweight synthetic experiment lab for prototyping data simulations and visualizing results quickly. It is useful for:

- exploring built-in synthetic experiment templates
- designing custom experiments with JSON model specs instead of Python code
- generating charts, metrics, dataset previews, and caption text in one flow
- running experiments through both CLI and web UI interfaces

## What it needs

The project is functional, but it still needs:

- an automated test suite for built-in experiments and custom JSON specs
- clearer JSON spec documentation and validation examples
- better error handling for invalid custom models
- packaging or distribution support for installation as a library
- performance tuning and benchmarking for larger dataset sizes
- additional built-in experiment templates and metric/chart coverage

## Contributing

Contributions are welcome. Good contributions include:

- adding unit or integration tests for synthetic dataset generation
- improving the custom model JSON schema and editor validation
- extending built-in experiments with new synthetic use cases
- refining the web UI experience and tab interaction behavior
- adding documentation, examples, and deployment notes

Please follow these basic guidelines:

- fork the repository and create a feature branch
- keep changes small and focused
- include tests or examples for new behavior
- document any public API or spec changes in the README

## Developer workflow

Install the project and test helpers with:

```bash
python -m pip install -e .[dev]
```

Run the test suite with:

```bash
python -m pytest
```

## License

This project is open source under the MIT License.

## Add A New Experiment

Create a new file in `internet_experiment_lab/experiments/` and define a `BaseExperiment` subclass:

```python
from internet_experiment_lab.core import BaseExperiment, ExperimentResult


class MyExperiment(BaseExperiment):
    name = "my_experiment"
    title = "My Experiment"

    def run(self, size=1000, seed=None):
        ...
        return ExperimentResult(...)
```

The engine discovers it automatically.
