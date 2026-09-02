"""Tests for the synthetic return generators.

Written before the implementation. The generators are the control group of
the whole experiment, so these tests check the two things that make a
control valid:

  * the series really has the distributional properties claimed for it,
    checked against values derived analytically rather than eyeballed;
  * the same seed always produces the same series, so every result
    regenerates exactly.

Where a quantity has a known closed form (the variance of a standardised
Student-t, the excess kurtosis of a GARCH(1,1) with normal innovations) the
test asserts against that form, not against a number recorded from an
earlier run.
"""

import numpy as np
import pytest
from scipy import stats

from datasnoop.synthetic import (
    garch11_excess_kurtosis_normal,
    garch11_log_returns,
    garch11_unconditional_variance,
    gbm_log_returns,
    generate_log_returns,
    prices_from_log_returns,
    standardised_t,
    student_t_log_returns,
)

# A large sample is needed to pin down a fourth moment; these are chosen so
# the assertions below hold comfortably rather than marginally.
N_MOMENT = 200_000
N_KURTOSIS = 400_000


def rng(seed=0):
    return np.random.default_rng(seed)


# --- reproducibility ------------------------------------------------------


def test_gbm_is_reproducible_from_its_seed():
    a = gbm_log_returns(1000, mu=0.0, sigma=0.01, rng=rng(7))
    b = gbm_log_returns(1000, mu=0.0, sigma=0.01, rng=rng(7))
    assert np.array_equal(a, b)


def test_different_seeds_give_different_series():
    a = gbm_log_returns(1000, mu=0.0, sigma=0.01, rng=rng(1))
    b = gbm_log_returns(1000, mu=0.0, sigma=0.01, rng=rng(2))
    assert not np.array_equal(a, b)


def test_garch_is_reproducible_from_its_seed():
    kw = dict(n=500, mu=0.0, omega=2e-6, alpha=0.08, beta=0.90)
    a = garch11_log_returns(**kw, rng=rng(7))
    b = garch11_log_returns(**kw, rng=rng(7))
    assert np.array_equal(a, b)


# --- geometric Brownian motion --------------------------------------------


def test_gbm_has_the_requested_length_and_moments():
    mu, sigma = 0.0005, 0.01
    r = gbm_log_returns(N_MOMENT, mu=mu, sigma=sigma, rng=rng(1))

    assert r.shape == (N_MOMENT,)
    # The sample mean of n draws has standard error sigma/sqrt(n); 4 SE is a
    # generous band that still fails loudly if mu or sigma is misapplied.
    assert abs(r.mean() - mu) < 4 * sigma / np.sqrt(N_MOMENT)
    assert abs(r.std(ddof=1) - sigma) < 4 * sigma / np.sqrt(2 * N_MOMENT)


def test_gbm_returns_are_gaussian():
    """Excess kurtosis of a normal is 0. This is the baseline the ladder starts from."""
    r = gbm_log_returns(N_KURTOSIS, mu=0.0, sigma=0.01, rng=rng(2))
    # SE of sample excess kurtosis under normality is sqrt(24/n).
    assert abs(stats.kurtosis(r, fisher=True, bias=True)) < 5 * np.sqrt(24 / N_KURTOSIS)


# --- prices ---------------------------------------------------------------


def test_prices_round_trip_to_the_log_returns_that_made_them():
    r = gbm_log_returns(500, mu=0.0003, sigma=0.02, rng=rng(3))
    p = prices_from_log_returns(r, p0=100.0)

    assert p.shape == (501,)
    assert p[0] == 100.0
    assert np.all(p > 0)
    np.testing.assert_allclose(np.log(p[1:] / p[:-1]), r, atol=1e-12)


def test_prices_stay_positive_under_extreme_negative_returns():
    """Exponentiating cannot produce a negative price. Note 1, section 4."""
    r = np.full(100, -0.5)
    assert np.all(prices_from_log_returns(r, p0=1.0) > 0)


# --- Student-t ------------------------------------------------------------


def test_standardised_t_has_unit_variance():
    """t_nu * sqrt((nu-2)/nu) has variance 1, so sigma keeps its meaning."""
    z = standardised_t(nu=6, size=N_MOMENT, rng=rng(4))
    assert abs(z.var(ddof=1) - 1.0) < 0.05


def test_standardised_t_matches_the_analytical_excess_kurtosis():
    """Excess kurtosis of a Student-t is 6/(nu-4) for nu > 4."""
    nu = 10  # far enough from 4 that the sample estimate is stable
    z = standardised_t(nu=nu, size=N_KURTOSIS, rng=rng(5))
    assert stats.kurtosis(z, fisher=True, bias=True) == pytest.approx(
        6 / (nu - 4), rel=0.15
    )


def test_standardised_t_tail_frequency_matches_the_distribution():
    """A direct probe of the tails, independent of any moment calculation."""
    nu = 8
    z = standardised_t(nu=nu, size=N_MOMENT, rng=rng(6))
    # Undo the scaling to compare against the raw t quantile.
    raw = z / np.sqrt((nu - 2) / nu)
    threshold = stats.t.ppf(0.99, nu)
    assert np.mean(np.abs(raw) > threshold) == pytest.approx(0.02, abs=0.003)


def test_student_t_returns_have_the_requested_scale():
    sigma = 0.01
    r = student_t_log_returns(N_MOMENT, mu=0.0, sigma=sigma, nu=6, rng=rng(7))
    assert r.std(ddof=1) == pytest.approx(sigma, rel=0.05)


@pytest.mark.parametrize("nu", [2.0, 1.5, 0.0, -1.0])
def test_student_t_rejects_degrees_of_freedom_without_a_finite_variance(nu):
    """Variance is infinite for nu <= 2, so sigma would be meaningless."""
    with pytest.raises(ValueError, match="nu"):
        standardised_t(nu=nu, size=10, rng=rng())


# --- GARCH(1,1) -----------------------------------------------------------


def test_garch_with_no_arch_or_garch_terms_reduces_to_gbm():
    """alpha = beta = 0 makes the conditional variance constant at omega.

    The series must then be identical to a GBM series with sigma = sqrt(omega)
    drawn from the same seed. This pins the recursion to the right model: any
    off-by-one or misplaced term shows up as a mismatch.
    """
    omega = 1e-4
    g = garch11_log_returns(
        1000, mu=0.0002, omega=omega, alpha=0.0, beta=0.0, rng=rng(8), burn_in=0
    )
    b = gbm_log_returns(1000, mu=0.0002, sigma=np.sqrt(omega), rng=rng(8))
    np.testing.assert_allclose(g, b, atol=1e-15)


def test_garch_variance_matches_the_unconditional_formula():
    """Long-run variance is omega / (1 - alpha - beta). Note 2, section 5."""
    omega, alpha, beta = 2e-6, 0.08, 0.90
    r = garch11_log_returns(
        N_MOMENT, mu=0.0, omega=omega, alpha=alpha, beta=beta, rng=rng(9)
    )
    expected = garch11_unconditional_variance(omega, alpha, beta)
    assert expected == pytest.approx(1e-4)
    assert r.var(ddof=1) == pytest.approx(expected, rel=0.10)


def test_garch_excess_kurtosis_matches_the_analytical_formula():
    """Clustering generates heavy tails even with Gaussian innovations.

    This is the claim in Note 2 that GARCH gives fat tails for free. The
    closed form for normal innovations is

        kurtosis = 3(1 - (a+b)^2) / (1 - (a+b)^2 - 2a^2)

    Parameters here are chosen well inside the region where the fourth
    moment exists, so the sample estimate is stable.
    """
    alpha, beta = 0.2, 0.5
    omega = 1e-4 * (1 - alpha - beta)
    r = garch11_log_returns(
        N_KURTOSIS, mu=0.0, omega=omega, alpha=alpha, beta=beta, rng=rng(10)
    )
    expected = garch11_excess_kurtosis_normal(alpha, beta)
    assert expected == pytest.approx(0.558, abs=0.01)
    assert stats.kurtosis(r, fisher=True, bias=True) == pytest.approx(
        expected, rel=0.20
    )


def test_garch_with_t_innovations_still_has_the_right_variance():
    omega, alpha, beta = 2e-6, 0.08, 0.90
    r = garch11_log_returns(
        N_MOMENT,
        mu=0.0,
        omega=omega,
        alpha=alpha,
        beta=beta,
        rng=rng(11),
        innovations="t",
        nu=8,
    )
    assert r.var(ddof=1) == pytest.approx(
        garch11_unconditional_variance(omega, alpha, beta), rel=0.20
    )


def test_garch_burn_in_does_not_change_the_output_length():
    r = garch11_log_returns(
        250, mu=0.0, omega=2e-6, alpha=0.08, beta=0.90, rng=rng(12), burn_in=500
    )
    assert r.shape == (250,)


@pytest.mark.parametrize("alpha,beta", [(0.5, 0.5), (0.6, 0.6), (1.0, 0.0)])
def test_garch_rejects_non_stationary_parameters(alpha, beta):
    """alpha + beta >= 1 has no finite long-run variance."""
    with pytest.raises(ValueError, match="stationar"):
        garch11_unconditional_variance(2e-6, alpha, beta)
    with pytest.raises(ValueError, match="stationar"):
        garch11_log_returns(10, mu=0.0, omega=2e-6, alpha=alpha, beta=beta, rng=rng())


@pytest.mark.parametrize("alpha,beta", [(-0.1, 0.5), (0.1, -0.5)])
def test_garch_rejects_negative_parameters(alpha, beta):
    """A negative coefficient could drive the variance below zero."""
    with pytest.raises(ValueError, match="non-negative"):
        garch11_log_returns(10, mu=0.0, omega=2e-6, alpha=alpha, beta=beta, rng=rng())


def test_garch_excess_kurtosis_formula_rejects_an_infinite_fourth_moment():
    """The formula requires 3a^2 + 2ab + b^2 < 1; otherwise kurtosis is infinite."""
    with pytest.raises(ValueError, match="fourth moment"):
        garch11_excess_kurtosis_normal(0.2, 0.9)


def test_garch_conditional_variance_responds_to_a_shock():
    """A large shock must raise the next period's variance, not lower it.

    Catches a sign error in the recursion, which a variance check alone
    would not: the unconditional variance can be right while the dynamics
    are backwards.
    """
    r = garch11_log_returns(
        20_000, mu=0.0, omega=2e-6, alpha=0.15, beta=0.80, rng=rng(13)
    )
    big = np.abs(r[:-1]) > np.quantile(np.abs(r[:-1]), 0.9)
    assert np.mean(r[1:][big] ** 2) > 2 * np.mean(r[1:][~big] ** 2)


def test_garch_parameters_are_recovered_by_an_independent_implementation():
    """Cross-check against the `arch` package by fitting a simulated series.

    The generator is written by hand so every line can be justified. This
    test uses an established implementation as an oracle: if `arch` fits a
    long series from our generator and recovers the parameters we put in,
    our recursion is the GARCH(1,1) the literature means.
    """
    arch = pytest.importorskip("arch")

    alpha, beta, omega = 0.10, 0.85, 1e-6
    r = garch11_log_returns(
        50_000, mu=0.0, omega=omega, alpha=alpha, beta=beta, rng=rng(14)
    )

    # `arch` fits better on numbers that are not tiny; scaling returns by a
    # constant scales omega by its square and leaves alpha and beta alone.
    result = arch.arch_model(r * 100, mean="Zero", vol="GARCH", p=1, q=1).fit(
        disp="off"
    )
    assert result.params["alpha[1]"] == pytest.approx(alpha, abs=0.02)
    assert result.params["beta[1]"] == pytest.approx(beta, abs=0.02)


# --- dispatch -------------------------------------------------------------


def test_generate_dispatches_to_each_model():
    specs = {
        "gbm": {"model": "gbm", "mu": 0.0, "sigma": 0.01},
        "student_t": {"model": "student_t", "mu": 0.0, "sigma": 0.01, "nu": 5},
        "garch11": {
            "model": "garch11",
            "mu": 0.0,
            "omega": 2e-6,
            "alpha": 0.08,
            "beta": 0.90,
        },
    }
    for spec in specs.values():
        assert generate_log_returns(100, spec, rng=rng(15)).shape == (100,)


def test_generate_matches_calling_the_generator_directly():
    spec = {"model": "gbm", "mu": 0.0, "sigma": 0.01}
    np.testing.assert_array_equal(
        generate_log_returns(50, spec, rng=rng(16)),
        gbm_log_returns(50, mu=0.0, sigma=0.01, rng=rng(16)),
    )


def test_generate_does_not_mutate_the_spec():
    """The spec comes from a config file and is written to the run log."""
    spec = {"model": "gbm", "mu": 0.0, "sigma": 0.01}
    before = dict(spec)
    generate_log_returns(10, spec, rng=rng())
    assert spec == before


def test_generate_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        generate_log_returns(10, {"model": "not_a_model"}, rng=rng())


def test_generate_rejects_a_spec_with_a_missing_parameter():
    """A typo in the config must fail loudly rather than use a default."""
    with pytest.raises((TypeError, KeyError)):
        generate_log_returns(10, {"model": "gbm", "mu": 0.0}, rng=rng())
