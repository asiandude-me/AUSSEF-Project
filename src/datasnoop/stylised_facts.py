"""Statistical tests for the stylised facts of financial returns.

A generator is only a valid control if the properties claimed for it are
measured rather than assumed. This module does the measuring. Three tests,
one per claim, from ``docs/notes/03_validation_tests.md``:

  * :func:`ljung_box` — is there autocorrelation? Applied twice, with
    *opposite* desired outcomes. On returns we want no autocorrelation (the
    series must be unpredictable in direction). On squared or absolute
    returns we want strong autocorrelation, because that is what volatility
    clustering is.
  * :func:`arch_lm` — Engle's test for volatility clustering specifically.
  * :func:`jarque_bera` — are the tails heavier than a normal's?

Each is written out from its formula in terms of numpy and scipy primitives
rather than imported from statsmodels. The project requires that every line
be defensible under questioning, and a formula written out is easier to
defend than a library call whose conventions have to be taken on trust.
That is only safe if the hand-written version is checked, so
``tests/test_stylised_facts.py`` asserts agreement with statsmodels and
scipy to 1e-8 on the same inputs.

Reading the output. A small p-value means reject the null. A large p-value
means the data did not provide evidence against the null — which is not the
same as proving it. The correct phrasing is "no significant autocorrelation
was detected", never "the series was proven to have none". And with a few
thousand observations an economically meaningless autocorrelation can still
reject, so :func:`summarise` reports effect sizes next to p-values.

References: Ljung & Box (1978), Engle (1982), Jarque & Bera (1980). See
``docs/references.md``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "arch_lm",
    "autocorrelation",
    "excess_kurtosis",
    "excess_kurtosis_se",
    "jarque_bera",
    "ljung_box",
    "summarise",
]


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation of `x` at lags ``1 .. max_lag``.

    For each lag ``k``, how similar is the series to itself shifted by ``k``::

        rho_k = sum_t (x_t - xbar)(x_{t-k} - xbar) / sum_t (x_t - xbar)^2

    Note the denominator uses all ``n`` terms while the numerator has only
    ``n - k``. That is the standard convention (statsmodels calls it
    ``adjusted=False``): it slightly shrinks estimates at long lags, which
    keeps the whole set of autocorrelations mutually consistent and is what
    the Ljung-Box statistic below assumes.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if max_lag < 1:
        raise ValueError(f"max_lag must be at least 1; got {max_lag}")
    if max_lag >= n:
        raise ValueError(f"max_lag ({max_lag}) must be less than n ({n})")

    centred = x - x.mean()
    denominator = np.dot(centred, centred)
    return np.array(
        [
            np.dot(centred[k:], centred[:-k]) / denominator
            for k in range(1, max_lag + 1)
        ]
    )


def ljung_box(x: np.ndarray, lags: int) -> tuple[float, float]:
    """Ljung-Box portmanteau test for autocorrelation up to ``lags``.

    H0: ``rho_1 = rho_2 = ... = rho_m = 0`` — no autocorrelation up to lag m.

    Any single sample autocorrelation is a little off zero by chance. Rather
    than peek at each one and pick the biggest (which is data snooping in
    miniature), the test combines them all into one statistic::

        Q = n(n + 2) * sum_{k=1..m} rho_k^2 / (n - k)

    Squaring means positive and negative correlations both count. The
    ``n(n+2)/(n-k)`` weighting is Ljung and Box's small-sample refinement of
    the earlier Box-Pierce statistic. Under H0, Q follows a chi-squared
    distribution with m degrees of freedom.

    Returns ``(Q, p_value)``. A small p-value means autocorrelation is
    present.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if lags >= n:
        raise ValueError(f"lags ({lags}) must be less than n ({n})")

    rho = autocorrelation(x, max_lag=lags)
    k = np.arange(1, lags + 1)
    q = n * (n + 2) * np.sum(rho**2 / (n - k))
    return float(q), float(stats.chi2.sf(q, df=lags))


def arch_lm(x: np.ndarray, lags: int) -> tuple[float, float]:
    """Engle's Lagrange multiplier test for ARCH effects (volatility clustering).

    H0: no ARCH effects — the squared series is not linearly predictable
    from its own past, i.e. the conditional variance is constant.

    The idea is direct. If volatility clusters then a big move today should
    be predictable from recent big moves, so regress ``x_t^2`` on its own m
    lags by ordinary least squares and ask whether that regression explains
    anything::

        LM = nobs * R^2

    where ``nobs`` is the number of rows in the regression (``n - m``) and
    ``R^2`` is its coefficient of determination. Under H0 this follows a
    chi-squared distribution with m degrees of freedom. The ``n * R^2`` form
    is the general shape of a Lagrange multiplier test, which is why it
    recurs throughout econometrics.

    Note that ``x`` is *not* demeaned before squaring, matching Engle's
    formulation and the statsmodels implementation this is checked against.

    Returns ``(LM, p_value)``. A small p-value means clustering is present.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if lags >= n:
        raise ValueError(f"lags ({lags}) must be less than n ({n})")

    squared = x**2
    response = squared[lags:]
    # Design matrix: a constant, then the m lagged values of the squared series.
    design = np.column_stack(
        [np.ones(response.size)]
        + [squared[lags - k : -k] for k in range(1, lags + 1)]
    )

    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ coefficients
    centred = response - response.mean()
    r_squared = 1.0 - np.dot(residuals, residuals) / np.dot(centred, centred)

    lm = response.size * r_squared
    return float(lm), float(stats.chi2.sf(lm, df=lags))


def _standardised_moment(x: np.ndarray, order: int) -> float:
    """The ``order``-th moment about the mean, divided by sd**order."""
    centred = x - x.mean()
    m2 = np.mean(centred**2)
    return float(np.mean(centred**order) / m2 ** (order / 2))


def jarque_bera(x: np.ndarray) -> tuple[float, float]:
    """Jarque-Bera test for normality, via skewness and kurtosis.

    H0: the data are normally distributed — specifically, they have the
    skewness (0) and kurtosis (3) of a normal.

    ::

        JB = (n / 6) * ( S^2 + (K - 3)^2 / 4 )

    where S is the sample skewness and K the sample kurtosis. Under H0 this
    follows a chi-squared distribution with **2** degrees of freedom: two,
    because two quantities are being tested at once.

    Returns ``(JB, p_value)``. A small p-value means reject normality.

    The convergence to the chi-squared limit is slow, so with a few hundred
    observations the test is conservative. That is a property of the
    asymptotics, not an error.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    skewness = _standardised_moment(x, 3)
    kurtosis = _standardised_moment(x, 4)
    jb = (n / 6.0) * (skewness**2 + (kurtosis - 3.0) ** 2 / 4.0)
    return float(jb), float(stats.chi2.sf(jb, df=2))


def excess_kurtosis(x: np.ndarray) -> float:
    """Sample excess kurtosis: the fourth standardised moment minus 3.

    Subtracting 3 puts a normal distribution at zero, so the number reads
    directly as "how much heavier than normal are the tails". Daily equity
    returns typically land somewhere around 3 to 10.

    Reported alongside the Jarque-Bera p-value because the p-value only says
    "not normal" while the magnitude is what can be compared against real
    market data.
    """
    return _standardised_moment(np.asarray(x, dtype=float), 4) - 3.0


def excess_kurtosis_se(n: int) -> float:
    """Approximate standard error of sample excess kurtosis under normality.

    ``sqrt(24 / n)``. For n = 2500 this is about 0.098, so an excess kurtosis
    of 4 sits roughly 40 standard errors from zero — which is also why the
    Jarque-Bera p-value is uninformative at that point and the magnitude is
    the useful output.
    """
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    return float(np.sqrt(24.0 / n))


def summarise(
    returns: np.ndarray, lags: list[int], significance_level: float
) -> dict[str, float | bool]:
    """Run every test on one return series and flatten the results.

    Returns a dict of plain Python floats and bools, so it can go straight
    into the run log or a pandas DataFrame.

    Ljung-Box is applied to three transformations of the series:

      * ``r`` — the returns. We want a *large* p-value: no predictable
        direction. This is the null-model requirement, and it must hold for
        every generator.
      * ``r2`` and ``abs_r`` — squared and absolute returns. We want *small*
        p-values on a GARCH series, because that is volatility clustering.

    Both ``r2`` and ``abs_r`` are reported. They measure the same thing, but
    ``r2`` needs a finite eighth moment for its usual null distribution,
    which fails for heavy-tailed innovations; ``abs_r`` is the more robust
    of the two and disagreement between them is itself informative.

    Effect sizes are included next to the p-values: with a few thousand
    observations a trivially small autocorrelation can be significant, so
    the lag-1 autocorrelations and the excess kurtosis are reported directly.
    """
    r = np.asarray(returns, dtype=float)
    transformations = {"r": r, "r2": r**2, "abs_r": np.abs(r)}

    out: dict[str, float | bool] = {}

    for lag in lags:
        for name, series in transformations.items():
            stat, p = ljung_box(series, lags=lag)
            out[f"lb_stat_{name}_lag{lag}"] = stat
            out[f"lb_p_{name}_lag{lag}"] = p
            out[f"lb_reject_{name}_lag{lag}"] = bool(p < significance_level)

        stat, p = arch_lm(r, lags=lag)
        out[f"arch_stat_lag{lag}"] = stat
        out[f"arch_p_lag{lag}"] = p
        out[f"arch_reject_lag{lag}"] = bool(p < significance_level)

    stat, p = jarque_bera(r)
    out["jb_stat"] = stat
    out["jb_p"] = p
    out["jb_reject"] = bool(p < significance_level)

    # Effect sizes, so a p-value is never reported on its own.
    out["excess_kurtosis"] = excess_kurtosis(r)
    out["excess_kurtosis_se"] = excess_kurtosis_se(r.size)
    for name, series in transformations.items():
        out[f"acf1_{name}"] = float(autocorrelation(series, max_lag=1)[0])
    out["mean"] = float(r.mean())
    out["std"] = float(r.std(ddof=1))

    return out
