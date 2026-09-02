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

### 2026-09-02 — continuous integration

- **CI runs the test suite** on every pull request and every push to `main`
  (`.github/workflows/tests.yml`). The point is evidence: the repository
  shows the tests passing at each stage of development, rather than only
  asserting that they did. That record is part of what the logbook and the
  commit history are meant to demonstrate.
- **CI installs from `requirements-lock.txt`, not by re-resolving
  dependencies.** The lock file records the exact versions the results were
  produced with, so the job checks that the *recorded* environment still
  installs and still passes. The package itself is then installed with
  `--no-deps`, because re-resolving at that point would defeat the lock.
  The known cost: CI will not notice a newer release of a dependency
  breaking the code. For a research repository that is the right trade,
  since matching the recorded environment matters more than tracking
  upstream. Revisit only if a dependency needs upgrading.
- **Python 3.11 in CI**, matching the version the lock file was frozen on.
  `Technical setup` allows 3.11+; the job verifies the recorded environment
  rather than the whole supported range.

### 2026-09-02 — synthetic generators and their validation

- **Module split.** `src/datasnoop/synthetic.py` makes series;
  `src/datasnoop/stylised_facts.py` measures them. Kept apart so the code
  that validates a generator cannot share a bug with the generator itself.
- **Units.** Every parameter is per trading day, in log-return units.
  `sigma = 0.01/day` is about 16% a year.
- **`mu = 0` in the validation config.** The stylised-facts tests are
  unaffected by a constant mean, and a zero-drift null is unambiguous: any
  positive Sharpe the later search reports is then purely an artefact of
  searching, with no drift-harvesting to argue about. **Still open:** the
  drift setting for the search experiment itself. Note 1 flags that a
  long-only rule earns `mu · E[s_t]` without skill, so the search must
  either use `mu = 0` or score every rule against buy-and-hold on the same
  series. Decide before running the search.
- **GARCH written by hand.** The recursion is five lines and is written out
  rather than delegated to `arch`, so every line can be justified. `arch` is
  used once, as an independent oracle: a test fits a GARCH(1,1) to 50,000
  points from our generator and confirms it recovers the parameters that
  produced them.
- **Burn-in of 500 steps**, started at the long-run variance, so the
  returned series is stationary from its first observation.
- **Student-t scaled to unit variance** by the factor `sqrt((nu-2)/nu)`, so
  `sigma` means the same thing — the unconditional daily standard deviation
  — for every generator. Without it, changing `nu` would silently change the
  volatility as well as the tails and confound the two. Requires `nu > 2`,
  which the code enforces.
- **GARCH parameters from the literature** (`alpha = 0.08, beta = 0.90`),
  not fitted to ASX data. Fitting would calibrate the control to the
  treatment — defensible, arguably better, but a different design — and it
  needs data that has not been downloaded yet. Recorded as a deliberate
  choice, not an oversight.
- **A fourth generator, GARCH with t innovations**, is validated alongside
  the three the design names. It costs one config entry and it is the row
  closest to real returns, so it informs which null the search should run on.
- **Statistics hand-written, then cross-checked.** Ljung-Box, ARCH-LM,
  Jarque-Bera and excess kurtosis are implemented from their formulas in
  numpy and scipy primitives. A test asserts they agree with statsmodels and
  scipy to 1e-8 on the same input. Defensible under questioning *and*
  checked against the reference implementation.
- **Independent series from one seed** via `SeedSequence(seed).spawn(S)`,
  which gives independent streams rather than one stream cut into pieces.
- **Rejection rates, never a single verdict.** Validation reports the
  fraction of series rejecting each test over `n_series = 500` independent
  series, with effect sizes beside every rate. A single series can pass or
  fail by luck.

#### The classical Ljung-Box test over-rejects on GARCH data

The first validation run rejected no-autocorrelation-in-returns on 26% of
GARCH series and 49% of GARCH-t series, against a nominal 5%. That column is
the null-model requirement, so this looked like a broken control.

It was the test, not the generator. Diagnosis: across 400 series the mean
sample autocorrelation was ~0 — the returns really are serially uncorrelated
— but its standard deviation was 24% (GARCH) and 49% (GARCH-t) above the
`1/sqrt(n)` the classical test assumes. Classical Ljung-Box needs the data
to be *independent* under the null; GARCH returns are uncorrelated but not
independent, so the assumed null variance is too small and the test treats
ordinary sampling noise as structure.

**Fix:** `robust_ljung_box` estimates each autocorrelation's variance from
the data instead of assuming it (Diebold 1986). Under independence the
estimate converges to 1 and the statistic reduces to Box-Pierce, so it
generalises the classical test rather than replacing it. The rejection rates
fall to 0.056 and 0.046, and an AR(1) injected into a GARCH series is still
detected at `p < 1e-4`.

**Both are reported**, classical and robust, rather than the classical one
being quietly dropped. The gap between them is a finding: it is a small
worked example of this project's own thesis, that a standard test applied
outside the assumptions it was derived under reports structure that is not
there. This belongs in the report.

#### Two effect sizes that look wrong and are not

- **GARCH excess kurtosis** measures ~0.80 at `n = 2500` against a closed
  form of 1.43. The fourth moment exists only when
  `3a² + 2ab + b² < 1`, satisfied here by 0.027, so convergence is slow:
  the median climbs to 1.41 by `n = 2,000,000` (checked). Ten years of daily
  data is too short to reach the asymptotic value. A consequence of choosing
  persistence realistic enough to matter, not a reason to change it.
- **Student-t excess kurtosis** at `nu = 5` is noisy across seeds, as Note 3
  predicted. Both are recorded in the results output where the numbers are
  seen, so the discrepancy is answered rather than discovered by a judge.

#### Run-log change

`git_dirty` now excludes `results/` as well as `logs/`. The first validation
run flagged itself dirty by writing its own figures and tables before the
log entry was composed, which defeats the flag: it asks whether the *code*
that produced a result was committed, not whether the run left files behind.
A test confirms an uncommitted source file alongside those outputs is still
seen.
