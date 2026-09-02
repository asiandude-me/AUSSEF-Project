"""Synthetic price series with no exploitable structure — the control group.

The experiment compares a strategy search on real market data against the
same search on data that provably contains no signal. This module makes that
second kind of data.

Three generators, in increasing order of realism:

  * ``gbm_log_returns`` — geometric Brownian motion. Independent normal
    log-returns. The simplest possible null.
  * ``student_t_log_returns`` — the same, with Student-t innovations. Adds
    heavy tails and nothing else.
  * ``garch11_log_returns`` — GARCH(1,1). Adds volatility clustering, and
    heavy tails come with it for free.

**Why all three are valid nulls.** A technical trading rule chooses its
position ``s_t`` from the past only, so its expected return is

    E[s_t * r_{t+1}] = E[s_t * E[r_{t+1} | past]] = mu * E[s_t]

which depends on the rule only through its average exposure, never through
its timing. Every generator here satisfies ``E[r_{t+1} | past] = mu``: in the
GARCH case the conditional *variance* moves around and is genuinely
forecastable, but the conditional *mean* is the constant ``mu``. Volatility
is predictable; direction is not. A crossover rule bets on direction, so it
has no edge on any of these series. See ``docs/notes/01_null_model.md`` and
``docs/notes/02_stylised_facts.md`` for the full argument.

**Everything is in log-returns.** Log-returns add across periods and are
scale-free, and exponentiating a cumulative sum can never produce a negative
price. ``prices_from_log_returns`` does that conversion.

**Randomness is explicit.** Every function takes a ``numpy.random.Generator``
and none touches global random state, so a series is reproducible from the
seed that built its generator.

References: Bollerslev (1986) for GARCH(1,1) and its fourth moment; Cont
(2001) for the stylised facts being reproduced. See ``docs/references.md``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "garch11_excess_kurtosis_normal",
    "garch11_log_returns",
    "garch11_unconditional_variance",
    "gbm_log_returns",
    "generate_log_returns",
    "prices_from_log_returns",
    "standardised_t",
    "student_t_log_returns",
]

DEFAULT_BURN_IN = 500


# --- innovations ----------------------------------------------------------


def standardised_t(nu: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw Student-t values rescaled to have variance exactly 1.

    A raw Student-t with ``nu`` degrees of freedom has variance ``nu/(nu-2)``,
    which is bigger than 1 and depends on ``nu``. Multiplying by
    ``sqrt((nu-2)/nu)`` brings the variance to 1. That matters because it lets
    ``sigma`` mean the same thing — the unconditional daily standard deviation
    — for every generator in this module. Without it, changing ``nu`` would
    silently change the volatility as well as the tails, and the comparison
    between generators would confound the two.

    The variance is finite only for ``nu > 2``, so smaller values are refused
    rather than returning something whose scale is undefined.
    """
    if nu <= 2:
        raise ValueError(
            f"nu must be greater than 2 for the variance to be finite; got {nu}"
        )
    return rng.standard_t(nu, size=size) * np.sqrt((nu - 2) / nu)


def _innovations(
    kind: str, size: int, rng: np.random.Generator, nu: float | None
) -> np.ndarray:
    """Draw ``size`` unit-variance innovations of the requested kind."""
    if kind == "normal":
        return rng.standard_normal(size)
    if kind == "t":
        if nu is None:
            raise ValueError("t innovations require nu")
        return standardised_t(nu, size, rng)
    raise ValueError(f"unknown innovations {kind!r}; expected 'normal' or 't'")


# --- generators -----------------------------------------------------------


def gbm_log_returns(
    n: int, mu: float, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Geometric Brownian motion, as ``n`` daily log-returns.

    In discrete time GBM is one line::

        r_t = mu + sigma * e_t,    e_t ~ N(0, 1), independent across t

    "Geometric" refers to what happens when these are exponentiated into
    prices: the log of the price takes a random walk, so the price itself
    moves multiplicatively and a given percentage move is the same size at
    any price level.

    ``mu`` is the mean daily log-return and ``sigma`` the daily volatility.
    Note that if you want prices to grow at expected simple rate ``m`` per
    day you must set ``mu = m - sigma**2 / 2``; the gap is Jensen's
    inequality, since averaging after exponentiating is not the same as
    exponentiating the average.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative; got {sigma}")
    return mu + sigma * rng.standard_normal(n)


def student_t_log_returns(
    n: int, mu: float, sigma: float, nu: float, rng: np.random.Generator
) -> np.ndarray:
    """Independent log-returns with Student-t innovations and heavy tails.

    Identical to :func:`gbm_log_returns` except that the innovation is a
    standardised Student-t instead of a normal. The returns are still
    independent, so there is still no predictable structure — only the shape
    of the distribution changes. Excess kurtosis is ``6/(nu-4)`` for
    ``nu > 4``; for smaller ``nu`` the fourth moment is infinite and the
    *sample* kurtosis grows with the sample size rather than settling down.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative; got {sigma}")
    return mu + sigma * standardised_t(nu, n, rng)


def garch11_unconditional_variance(omega: float, alpha: float, beta: float) -> float:
    """Long-run variance of a GARCH(1,1) process, ``omega / (1 - alpha - beta)``.

    Taking expectations of the variance recursion and setting both sides
    equal to the same constant ``v`` gives ``v = omega + alpha*v + beta*v``,
    hence ``v = omega / (1 - alpha - beta)``. That is finite and positive
    only when ``alpha + beta < 1``, which is the stationarity condition: the
    closer the sum is to 1, the more persistent the volatility clustering.
    """
    _check_garch_parameters(omega, alpha, beta)
    return omega / (1.0 - alpha - beta)


def garch11_excess_kurtosis_normal(alpha: float, beta: float) -> float:
    """Excess kurtosis of a GARCH(1,1) with *normal* innovations.

    Bollerslev (1986). Writing ``s = alpha + beta``::

        kurtosis = 3 * (1 - s**2) / (1 - s**2 - 2*alpha**2)

    and excess kurtosis is that minus 3. The result is strictly positive
    whenever ``alpha > 0``, which is the point worth understanding: the
    observed return is ``sigma_t * e_t`` with ``sigma_t`` varying over time,
    and mixing normals of different variances puts more mass in the tails
    than any single normal has. Volatility clustering *generates* heavy
    tails. Two stylised facts, one cause.

    The fourth moment is finite only when ``3*alpha**2 + 2*alpha*beta +
    beta**2 < 1``; outside that region the kurtosis is infinite and this
    function raises rather than returning a meaningless negative number.
    """
    _check_garch_parameters(1.0, alpha, beta)
    if 3 * alpha**2 + 2 * alpha * beta + beta**2 >= 1:
        raise ValueError(
            "fourth moment is infinite for these parameters: "
            "3*alpha^2 + 2*alpha*beta + beta^2 must be < 1"
        )
    s = alpha + beta
    return 3 * (1 - s**2) / (1 - s**2 - 2 * alpha**2) - 3


def _check_garch_parameters(omega: float, alpha: float, beta: float) -> None:
    if omega <= 0:
        raise ValueError(f"omega must be positive; got {omega}")
    if alpha < 0 or beta < 0:
        raise ValueError(
            f"alpha and beta must be non-negative so the variance cannot go "
            f"below zero; got alpha={alpha}, beta={beta}"
        )
    if alpha + beta >= 1:
        raise ValueError(
            f"alpha + beta must be < 1 for the variance to be stationary; "
            f"got alpha + beta = {alpha + beta}"
        )


def garch11_log_returns(
    n: int,
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
    *,
    innovations: str = "normal",
    nu: float | None = None,
    burn_in: int = DEFAULT_BURN_IN,
) -> np.ndarray:
    """GARCH(1,1) log-returns: volatility that clusters, direction that does not.

    The model, in full::

        r_t     = mu + a_t
        a_t     = sigma_t * e_t,                  e_t unit-variance, independent
        sigma_t^2 = omega + alpha * a_{t-1}^2 + beta * sigma_{t-1}^2

    Read the last line in words: today's variance is a baseline ``omega``,
    plus a reaction to how big yesterday's move was (``alpha``), plus a
    memory of yesterday's variance (``beta``). A large move raises the
    variance, which makes another large move more likely, which is exactly
    what volatility clustering means. ``alpha + beta`` controls persistence;
    typical fitted daily equity values are around ``alpha = 0.05-0.1`` and
    ``beta = 0.85-0.9``.

    Crucially the conditional *mean* is still ``mu``, because ``sigma_t`` is
    fixed by the past and ``e_t`` has mean zero. The series is unpredictable
    in direction, so it remains a valid null for a directional trading rule.

    The recursion starts at the long-run variance and then discards the first
    ``burn_in`` values, so the returned series is stationary from its first
    observation rather than carrying a transient from the starting value.

    All innovations are drawn as a single array before the loop. That keeps
    the draws in one deterministic sequence, so ``alpha = beta = 0`` with no
    burn-in reproduces :func:`gbm_log_returns` exactly from the same seed —
    which is how the tests confirm the recursion implements the right model.
    """
    _check_garch_parameters(omega, alpha, beta)
    if burn_in < 0:
        raise ValueError(f"burn_in must be non-negative; got {burn_in}")

    total = n + burn_in
    e = _innovations(innovations, total, rng, nu)

    a = np.empty(total)
    variance = garch11_unconditional_variance(omega, alpha, beta)
    for t in range(total):
        a[t] = np.sqrt(variance) * e[t]
        variance = omega + alpha * a[t] ** 2 + beta * variance

    return mu + a[burn_in:]


# --- prices ---------------------------------------------------------------


def prices_from_log_returns(
    log_returns: np.ndarray, p0: float = 100.0
) -> np.ndarray:
    """Turn ``n`` log-returns into ``n + 1`` prices starting at ``p0``.

    ``P_t = P_0 * exp(r_1 + ... + r_t)``. The output is one longer than the
    input because it includes the starting price. Because ``exp`` of anything
    is positive, a price built this way can never go negative — the reason
    the whole project works in log-returns rather than putting noise directly
    on the price level.
    """
    if p0 <= 0:
        raise ValueError(f"p0 must be positive; got {p0}")
    log_returns = np.asarray(log_returns, dtype=float)
    return p0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))


# --- dispatch -------------------------------------------------------------


def generate_log_returns(
    n: int, spec: dict, rng: np.random.Generator
) -> np.ndarray:
    """Generate ``n`` log-returns from a config dict.

    ``spec`` carries a ``model`` key naming the generator and the remaining
    keys are passed through as its arguments, so the config file is the only
    place any parameter value appears. An unknown model, or a missing
    parameter, raises rather than falling back to a default: a typo in a
    config must fail loudly, not silently produce a series nobody asked for.

    The spec is not mutated, since the same dict is written to the run log.
    """
    generators = {
        "gbm": gbm_log_returns,
        "student_t": student_t_log_returns,
        "garch11": garch11_log_returns,
    }
    params = dict(spec)
    model = params.pop("model", None)
    if model not in generators:
        raise ValueError(
            f"unknown model {model!r}; expected one of {sorted(generators)}"
        )
    return generators[model](n, rng=rng, **params)
