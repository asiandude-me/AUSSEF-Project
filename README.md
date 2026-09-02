# Measuring data-snooping bias in technical trading strategy search

A controlled computational experiment. A systematic search over technical
trading rules (moving-average crossovers over a large parameter grid) is run
on synthetic price series that contain no exploitable structure by
construction, to build an empirical distribution of the best in-sample
Sharpe ratio obtainable from pure noise. The same search is then run on real
market data and the result is compared against that noise baseline.

The project is not an attempt to build a profitable trading strategy. See
`CLAUDE.md` for the full research design, constraints, and the log of design
decisions.

## Setup

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Exact package versions used during development are pinned in
`requirements-lock.txt`. To reproduce that environment exactly:

```
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

## Tests

```
pytest -q
```

Every function that does statistical work has a test against a known
analytical answer. Statistical routines are written out from their formulas
rather than imported, so that each can be justified without relying on a
library's conventions; tests then assert agreement with `statsmodels` and
`scipy` to 1e-8 on the same inputs.

## Layout

```
src/datasnoop/   library code (installed package)
experiments/     run scripts; each one appends to the run log
configs/         YAML configuration files for experiments
tests/           pytest test suite
docs/            references and background notes
logs/            run log (logs/runs.jsonl), committed to git
results/         generated figures and tables
data/            downloaded market data (gitignored)
notebooks/       exploration only; nothing here is a result
```

Directories are created when the first component that uses them is built.

## Experiments

| Script | What it does |
| --- | --- |
| `experiments/validate_synthetic.py` | Validates the synthetic generators against the stylised facts of real returns, writing `results/synthetic_validation/`. No generator is used as a control until it passes. |

## Run log

Every experiment run appends one JSON object to `logs/runs.jsonl` through
the `RunLog` context manager in `src/datasnoop/runlog.py`. Each line records
the UTC start and finish time, runtime, git commit hash, whether the working
tree was dirty, the full config, the random seed, key outputs, and whether
the run succeeded or failed. The file is append-only and the logger has no
way to set timestamps, so entries are contemporaneous by construction.

Standard procedure: commit code, run the experiment, commit the new log line.

```python
from datasnoop.runlog import RunLog

with RunLog(experiment="name", config=CONFIG, seed=SEED) as run:
    ...
    run.record(best_sharpe=value)
```
