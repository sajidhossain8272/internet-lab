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
```

The same `app.py` also exposes a top-level WSGI `app` callable for Vercel. The deployed web routes are:

- `/` for the experiment launcher
- `/run/economy`, `/run/attention`, `/run/jobs`, `/run/passwords`, `/run/behavior`
- `/api/run/<experiment>` for JSON output

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
  experiments/
    economy.py
    social_attention.py
    job_market.py
    password_strength.py
    human_behavior.py
```

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
