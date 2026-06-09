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

On the Vercel homepage, the lab now behaves like an interactive runner: select a preset experiment or edit a custom model spec, set the row count and seed, watch the terminal-style progress log, then see the generated result page, SVG charts, metrics, tweet caption, and dataset preview appear dynamically.

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
