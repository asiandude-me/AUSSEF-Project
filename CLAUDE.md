# Project: Measuring data-snooping bias in technical trading strategy search

## What this is

A computational science research project for submission to the Australian
Science and Engineering Fair (AUSSEF), category: Mathematics (MATH) or
Systems Software (SOFT). Submission deadline 11 November 2026. It will be
judged against an ISEF-derived rubric and, if shortlisted, defended in a
10-15 minute live interview.

## The research question

When you search a large space of technical trading rules over a price
series, how much apparent predictive signal do you find in data that
provably contains none — and is the signal found in real market data
distinguishable from that noise baseline?

## The core design (do not deviate from this without asking)

This is a controlled experiment. The control group is synthetic price data
with zero predictability by construction. The treatment group is real market
data. The same strategy-search pipeline runs over both.

1. Generate synthetic price series with no exploitable structure:
   - Geometric Brownian motion (baseline)
   - Student-t innovations (fat tails)
   - A GARCH(1,1) process (volatility clustering)
   Each variant must be validated against the stylised facts of real returns
   before it is used: near-zero autocorrelation of returns, significant
   autocorrelation of squared/absolute returns, excess kurtosis.

2. Run an identical systematic strategy search over each series. Start with
   moving-average crossover rules over a parameter grid (fast period x slow
   period x holding rule), on the order of 10^3-10^4 combinations. Record
   the best in-sample Sharpe ratio found per series.

3. Repeat over many independent synthetic series to build an empirical
   distribution of "best Sharpe obtainable from pure noise."

4. Run the same pipeline over real ASX data (index and a sample of
   constituents, multiple time periods).

5. Compare: does the best real-data Sharpe fall outside the noise
   distribution, or inside it? Report a p-value against the synthetic null.

6. Hold out data. Re-test the in-sample winners out-of-sample and quantify
   the degradation.

7. Apply and compare established multiple-testing corrections — at minimum
   White's Reality Check and the Deflated Sharpe Ratio (Bailey & Lopez de
   Prado). Show how each changes the conclusion.

## What this project is NOT

It is not an attempt to build a profitable trading strategy. If any
suggestion drifts toward "improve returns", "optimise the strategy", or
"add machine learning to predict prices", stop and flag it. The finding
that strategies fail is the result, not a failure of the project.

Do not add a machine-learning return predictor. Do not report backtest
returns as if they were evidence of skill.

## Non-negotiable constraints

**Defensibility over sophistication.** I have to explain every line of this
under questioning by research scientists. If there is a simple method and a
sophisticated one, take the simple one unless the sophisticated one is
necessary and you have explained it to me until I can reproduce the
explanation myself. No library call I cannot justify.

**Reproducibility.** Every result must regenerate exactly from a seed and a
config file. No hardcoded magic numbers in analysis code. Cache downloaded
market data to disk so results do not change when the data source updates.

**Audit trail.** Every experiment run must append to a structured run log:
UTC timestamp, git commit hash, full config, random seed, key outputs,
runtime. This feeds a mandatory research logbook, so it must be genuine and
contemporaneous. Never backdate or fabricate a log entry.

**Anonymity.** The submission is blind-judged. No personal name, school
name, teacher name, state or city may appear anywhere in the repository —
including code comments, docstrings, commit messages, file names, plot
titles, or output paths.

**Honest attribution.** This project builds on existing literature
(data-snooping bias, White 2000, Bailey & Lopez de Prado 2014, Harvey &
Liu). Maintain a references file. Where a method comes from a paper, say
so in the code comment and in the writeup. Concealed derivation is
treated far more harshly by judges than acknowledged derivation.

## Technical setup

- Python 3.11+, numpy, pandas, scipy, statsmodels, arch (for GARCH),
  matplotlib. Ask before adding anything else.
- Structure: `src/` for library code, `experiments/` for run scripts,
  `configs/` for YAML configs, `data/raw/` and `data/cache/` (gitignored),
  `results/`, `logs/`, `notebooks/` for exploration only.
- pytest for tests. Any function doing statistical work needs a test with a
  known analytical answer.
- Market data: yfinance with `.AX` suffixed tickers. Check and note the
  terms of use. Cache everything locally on first fetch.

## How I want you to work

- One step at a time. Do not scaffold the entire project in one go.
- Before writing code for a new component, explain the approach in plain
  language and check I have followed it. I am a Year 12 student; I know
  Python and senior maths, I do not yet know time-series econometrics.
- Prefer explaining a statistical concept over silently importing a
  function that implements it.
- When I ask for something that would weaken the experimental design, say
  so directly rather than complying.
- Write the tests before the implementation where practical.

## Design decisions log

Record every design decision here as it is made, with the date. By the end
of the project this section is the skeleton of the methodology section.

### 2026-09-02 — project setup and run logging

- **Package layout.** Library code lives in `src/datasnoop/` (src layout,
  installed with `pip install -e .`), so tests and experiment scripts import
  the same installed package. Package name is neutral and descriptive; it
  carries no identifying information.
- **Run log format.** Append-only JSON Lines at `logs/runs.jsonl`, one JSON
  object per line, one line per experiment run. Chosen over a database or
  CSV because it is human-readable, diff-friendly in git, trivially parsed
  by `json` and pandas, and appending a line cannot corrupt earlier lines.
  The file is committed to git so the audit trail travels with the code.
- **Run log fields.** `schema_version, experiment, started_utc,
  finished_utc, runtime_seconds, git_commit, git_dirty, python_version,
  seed, config, outputs, status, error`. Timestamps are taken by the
  machine inside a context manager; the API has no argument to set them,
  so entries cannot be backdated through the logger. Failed runs are logged
  with `status: "failed"` and the exception text, then the exception is
  re-raised. Broken runs are part of the record.
- **Fields deliberately excluded from the run log.** Hostname, username,
  home directory, absolute paths. Any of these can leak a personal or
  machine name into the repository, which the anonymity rule forbids. A
  unit test enforces the exclusion.
- **`git_dirty` semantics.** True if `git status --porcelain` reports any
  change outside `logs/`. The log file itself is excluded so that the
  previous run's log line does not mark the next run as dirty. Untracked
  source files do count as dirty, because a result produced by uncommitted
  code is not reproducible from the recorded commit hash.
- **Standard run procedure.** Commit code, then run the experiment, then
  commit the new log line. This keeps `git_dirty` false and the recorded
  commit hash equal to the code that produced the result.
- **Dependencies.** `pyyaml` added to the list in Technical setup, because
  YAML config files need a parser and there is no YAML parser in the
  standard library. Exact versions used are pinned in
  `requirements-lock.txt` from the environment where tests were run.
- **Background notes.** Concept explanations produced during the build with
  AI assistance live in `docs/notes/` and are labelled as such. They are
  study material, not part of the submitted report, and the report must be
  written independently of them.
