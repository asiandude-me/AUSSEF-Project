"""Validate the synthetic generators against the stylised facts of real returns.

Step 1 of the core design: no generator may be used as a control until the
properties claimed for it have been *measured*. This script does the
measuring and writes the evidence to `results/synthetic_validation/`.

What it does. For each generator in the config, it builds `n_series`
independent series, runs every test in `datasnoop.stylised_facts` on each,
and reports the **rejection rate** over those series rather than one series'
verdict. A single series can pass or fail by luck; a rate over hundreds
cannot. Effect sizes are reported next to every rate, because with 2500
observations a trivially small autocorrelation can still be significant.

What to look for in the output, from docs/notes/03_validation_tests.md:

    Generator      p_LB(r)   p_LB(r^2)  p_ARCH   p_JB    excess kurtosis
    GBM            large     large      large    large   ~ 0
    Student-t      large     large      large    tiny    large, positive
    GARCH(1,1)     large     tiny       tiny     tiny    positive

so the rejection *rates* should be ~ the significance level wherever the
table says "large", and ~ 1 wherever it says "tiny".

The leftmost column is the one that matters most. For every generator the
returns themselves must show no significant autocorrelation, at close to the
nominal rate. That is the null-model requirement: if a generator fails it,
it is not a valid control and nothing downstream can be trusted.

Run from anywhere in the repository:

    python experiments/validate_synthetic.py
"""

from pathlib import Path

import matplotlib

# A non-interactive backend, so the script runs the same way with or without
# a display attached.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from datasnoop.runlog import RunLog, find_repo_root
from datasnoop.stylised_facts import autocorrelation, summarise
from datasnoop.synthetic import generate_log_returns

CONFIG_RELPATH = Path("configs") / "synthetic_validation.yaml"
OUTPUT_RELDIR = Path("results") / "synthetic_validation"

# Validated for colour-vision deficiency with the data-visualisation
# palette checker: worst-pair separation dE 24.7, well above the target of 8.
COLOUR_RETURNS = "#2a78d6"
COLOUR_SQUARED = "#eb6834"
COLOUR_BAND = "#b8b7b0"


def independent_generators(seed, n_series):
    """One independent random generator per series.

    `SeedSequence.spawn` produces independent streams rather than cutting a
    single stream into pieces, which is what "independent series" has to mean
    for a rejection rate over them to be meaningful. Everything still derives
    from the one seed in the config, so the whole run reproduces exactly.
    """
    return [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(n_series)]


def run_one_generator(spec, config, seed):
    """Return per-series test results and the mean correlograms."""
    n_days = config["n_days"]
    max_lag = config["acf_max_lag"]

    rows = []
    acf_returns = []
    acf_squared = []

    for rng in independent_generators(seed, config["n_series"]):
        returns = generate_log_returns(n_days, spec, rng=rng)
        rows.append(
            summarise(
                returns,
                lags=config["lags"],
                significance_level=config["significance_level"],
            )
        )
        acf_returns.append(autocorrelation(returns, max_lag=max_lag))
        acf_squared.append(autocorrelation(returns**2, max_lag=max_lag))

    return (
        pd.DataFrame(rows),
        np.mean(acf_returns, axis=0),
        np.mean(acf_squared, axis=0),
    )


def aggregate(name, results, lags):
    """Collapse one generator's per-series results into a single row."""
    primary = lags[0]
    row = {"generator": name, "n_series": len(results)}

    for lag in lags:
        for series in ("r", "r2", "abs_r"):
            row[f"reject_lb_{series}_lag{lag}"] = results[
                f"lb_reject_{series}_lag{lag}"
            ].mean()
        row[f"reject_arch_lag{lag}"] = results[f"arch_reject_lag{lag}"].mean()

    row["reject_jb"] = results["jb_reject"].mean()

    # Effect sizes, so no rate is reported without the magnitude behind it.
    row["median_excess_kurtosis"] = results["excess_kurtosis"].median()
    row["iqr_excess_kurtosis"] = (
        results["excess_kurtosis"].quantile(0.75)
        - results["excess_kurtosis"].quantile(0.25)
    )
    row["mean_acf1_r"] = results["acf1_r"].mean()
    row["mean_acf1_r2"] = results["acf1_r2"].mean()
    row["median_std"] = results["std"].median()
    row[f"median_p_lb_r_lag{primary}"] = results[f"lb_p_r_lag{primary}"].median()
    row[f"median_p_lb_r2_lag{primary}"] = results[f"lb_p_r2_lag{primary}"].median()
    row[f"median_p_arch_lag{primary}"] = results[f"arch_p_lag{primary}"].median()
    row["median_p_jb"] = results["jb_p"].median()
    return row


def acceptance_table(summary, config):
    """Note 3's acceptance table, with the measured numbers in the cells."""
    primary = config["lags"][0]
    level = config["significance_level"]
    n_series = config["n_series"]
    # A correctly calibrated rate is Binomial(n_series, level) / n_series.
    se = np.sqrt(level * (1 - level) / n_series)

    lines = [
        "# Synthetic generator validation",
        "",
        f"{n_series} independent series per generator, {config['n_days']} days each, "
        f"tested at the {level:g} level with Ljung-Box and ARCH-LM at lag {primary}.",
        "",
        "Each cell is the **fraction of series rejecting** that test. Where the",
        "acceptance table in `docs/notes/03_validation_tests.md` says the p-value",
        f"should be large, a correct generator gives a rate near {level:g}; where it",
        "says tiny, a rate near 1.",
        "",
        f"A correctly calibrated rate has standard error {se:.4f}, so anything in",
        f"roughly {level - 3 * se:.3f} to {level + 3 * se:.3f} is consistent with {level:g}.",
        "",
        "| Generator | reject LB(r) | reject LB(r²) | reject LB(&#124;r&#124;) | reject ARCH | reject JB | median excess kurtosis (IQR) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['generator']} "
            f"| {row[f'reject_lb_r_lag{primary}']:.3f} "
            f"| {row[f'reject_lb_r2_lag{primary}']:.3f} "
            f"| {row[f'reject_lb_abs_r_lag{primary}']:.3f} "
            f"| {row[f'reject_arch_lag{primary}']:.3f} "
            f"| {row['reject_jb']:.3f} "
            f"| {row['median_excess_kurtosis']:.2f} ({row['iqr_excess_kurtosis']:.2f}) |"
        )

    lines += [
        "",
        "## How to read the first column",
        "",
        "`reject LB(r)` is the null-model requirement. Every generator must sit",
        f"near {level:g} here: the returns themselves carry no predictable direction,",
        "which is what makes the series a valid control for a directional trading",
        "rule. A generator failing this column is broken as a control, and that is",
        "a correctness bug rather than a cosmetic one.",
        "",
        "The remaining columns are what separates the generators, and they are",
        "meant to differ: heavy tails show up in `reject JB`, volatility clustering",
        "in `reject LB(r²)` and `reject ARCH`.",
        "",
        "## Rejection rates at every lag",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
    ]
    return "\n".join(lines)


def write_figure(correlograms, config, path):
    """Correlograms: ACF of returns and of squared returns, per generator.

    Small multiples, because the comparison being made is the *shape* of the
    decay across generators. Each row is a generator; the left column is the
    autocorrelation of returns and the right column of squared returns. The
    shaded band is the +/- 1.96/sqrt(n) region within which an individual
    autocorrelation is not significantly different from zero.

    The expected picture, and the whole point of the figure: the left column
    is flat and inside the band for every generator, while the right column
    is flat only for GBM and Student-t and decays slowly for the two GARCH
    rows. Direction is unpredictable everywhere; volatility is not.
    """
    n_days = config["n_days"]
    max_lag = config["acf_max_lag"]
    lags = np.arange(1, max_lag + 1)
    band = 1.96 / np.sqrt(n_days)

    names = list(correlograms)
    fig, axes = plt.subplots(
        len(names), 2, figsize=(9.5, 2.05 * len(names)), sharex=True, sharey=True
    )

    for row, name in enumerate(names):
        for col, (values, colour, label) in enumerate(
            [
                (correlograms[name][0], COLOUR_RETURNS, "returns"),
                (correlograms[name][1], COLOUR_SQUARED, "squared returns"),
            ]
        ):
            ax = axes[row, col]
            ax.axhspan(-band, band, color=COLOUR_BAND, alpha=0.35, linewidth=0)
            ax.axhline(0, color="#52514e", linewidth=0.8)
            # Thin stems: the standard correlogram mark, one per lag.
            ax.vlines(lags, 0, values, color=colour, linewidth=1.4)
            ax.set_ylabel(name if col == 0 else "")
            if row == 0:
                ax.set_title(f"autocorrelation of {label}", fontsize=10)
            if row == len(names) - 1:
                ax.set_xlabel("lag (trading days)")
            ax.grid(axis="y", color="#e6e5df", linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

    fig.suptitle(
        "Synthetic generators: direction is unpredictable, volatility is not",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.005,
        f"Mean over {config['n_series']} independent series of {n_days} days. "
        f"Shaded band is ±1.96/√n, the region where a single "
        f"autocorrelation is not significant.",
        ha="center",
        fontsize=8,
        color="#52514e",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    repo_root = find_repo_root() or Path.cwd()
    config = yaml.safe_load((repo_root / CONFIG_RELPATH).read_text())
    output_dir = repo_root / OUTPUT_RELDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with RunLog(
        experiment="validate_synthetic", config=config, seed=config["seed"]
    ) as run:
        summary_rows = []
        correlograms = {}

        for index, (name, spec) in enumerate(config["generators"].items()):
            # A distinct seed per generator, derived from the one config seed,
            # so the generators do not share a random stream.
            results, acf_r, acf_r2 = run_one_generator(
                spec, config, seed=config["seed"] + index
            )
            summary_rows.append(aggregate(name, results, config["lags"]))
            correlograms[name] = (acf_r, acf_r2)

        summary = pd.DataFrame(summary_rows)

        csv_path = output_dir / "summary.csv"
        markdown_path = output_dir / "summary.md"
        figure_path = output_dir / "correlograms.png"

        summary.to_csv(csv_path, index=False)
        markdown_path.write_text(acceptance_table(summary, config) + "\n")
        write_figure(correlograms, config, figure_path)

        primary = config["lags"][0]
        run.record(
            outputs_dir=str(OUTPUT_RELDIR),
            n_series_per_generator=config["n_series"],
            # The headline numbers, so the log line is readable on its own
            # without opening the results files.
            rejection_rate_ljung_box_returns={
                row["generator"]: round(row[f"reject_lb_r_lag{primary}"], 4)
                for row in summary_rows
            },
            rejection_rate_arch={
                row["generator"]: round(row[f"reject_arch_lag{primary}"], 4)
                for row in summary_rows
            },
            rejection_rate_jarque_bera={
                row["generator"]: round(row["reject_jb"], 4) for row in summary_rows
            },
            median_excess_kurtosis={
                row["generator"]: round(row["median_excess_kurtosis"], 4)
                for row in summary_rows
            },
        )

        print(markdown_path.read_text())


if __name__ == "__main__":
    main()
