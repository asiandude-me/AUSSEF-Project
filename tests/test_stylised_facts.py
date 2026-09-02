"""Tests for the stylised-facts statistics.

Written before the implementation. These functions are hand-written from the
formulas in ``docs/notes/03_validation_tests.md`` rather than imported, so
that every line can be justified under questioning. That choice is only safe
if the hand-written versions are checked against established implementations,
which is what the first group of tests does: agreement with statsmodels and
scipy to 1e-8 on the same input.

The second group checks the statistics against analytical answers (the
autocorrelation of an AR(1), the kurtosis of a two-point distribution).

The third is the calibration check from Note 3, section 6, and is the most
valuable test in the file: if the null hypothesis is true, a p-value is
uniform on [0, 1], so the rejection rate at level alpha must be alpha. The
tolerance is computed from the binomial standard error inside the test rather
than hardcoded.
"""

import numpy as np
import pytest
from scipy import stats

from datasnoop.stylised_facts import (
    arch_lm,
    autocorrelation,
    excess_kurtosis,
    excess_kurtosis_se,
    jarque_bera,
    ljung_box,
    summarise,
)
from datasnoop.synthetic import garch11_log_returns, gbm_log_returns, standardised_t

TOLERANCE = 1e-8


def rng(seed=0):
    return np.random.default_rng(seed)


def ar1(n, phi, rng, sigma=1.0):
    """An AR(1) series: x_t = phi * x_{t-1} + noise. Autocorrelated by design."""
    x = np.empty(n)
    x[0] = rng.standard_normal() * sigma / np.sqrt(1 - phi**2)
    noise = rng.standard_normal(n) * sigma
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    return x


# --- agreement with the reference implementations -------------------------


def test_autocorrelation_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.tsa.stattools")
    x = rng(1).standard_normal(500)
    np.testing.assert_allclose(
        autocorrelation(x, max_lag=20),
        sm.acf(x, nlags=20, adjusted=False)[1:],
        atol=TOLERANCE,
    )


def test_ljung_box_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.diagnostic")
    x = rng(2).standard_normal(500)
    for lag in (10, 20, 50):
        stat, p = ljung_box(x, lags=lag)
        reference = sm.acorr_ljungbox(x, lags=[lag], return_df=True)
        assert stat == pytest.approx(reference["lb_stat"].iloc[0], abs=TOLERANCE)
        assert p == pytest.approx(reference["lb_pvalue"].iloc[0], abs=TOLERANCE)


def test_arch_lm_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.diagnostic")
    x = rng(3).standard_normal(500)
    for lag in (5, 10, 20):
        stat, p = arch_lm(x, lags=lag)
        ref_stat, ref_p = sm.het_arch(x, nlags=lag, result_object=False)[:2]
        assert stat == pytest.approx(ref_stat, abs=TOLERANCE)
        assert p == pytest.approx(ref_p, abs=TOLERANCE)


def test_jarque_bera_matches_scipy():
    for seed in (4, 5, 6):
        x = rng(seed).standard_normal(500)
        stat, p = jarque_bera(x)
        reference = stats.jarque_bera(x)
        assert stat == pytest.approx(reference.statistic, abs=TOLERANCE)
        assert p == pytest.approx(reference.pvalue, abs=TOLERANCE)


def test_excess_kurtosis_matches_scipy():
    x = rng(7).standard_normal(500)
    assert excess_kurtosis(x) == pytest.approx(
        stats.kurtosis(x, fisher=True, bias=True), abs=TOLERANCE
    )


# --- analytical answers ---------------------------------------------------


def test_autocorrelation_of_an_ar1_decays_geometrically():
    """An AR(1) with parameter phi has autocorrelation phi**k at lag k."""
    phi = 0.5
    x = ar1(200_000, phi, rng(8))
    rho = autocorrelation(x, max_lag=4)
    np.testing.assert_allclose(rho, [phi**k for k in range(1, 5)], atol=0.02)


def test_autocorrelation_of_white_noise_is_near_zero():
    """Sample autocorrelation of independent draws has SE about 1/sqrt(n)."""
    n = 100_000
    rho = autocorrelation(rng(9).standard_normal(n), max_lag=10)
    assert np.all(np.abs(rho) < 5 / np.sqrt(n))


def test_excess_kurtosis_of_a_two_point_distribution_is_exactly_minus_two():
    """For values at +/-1 with equal weight, m4 = 1 and m2 = 1, so kurtosis
    is 1 and excess kurtosis is exactly -2. An exact answer, no sampling."""
    x = np.array([-1.0, 1.0] * 50)
    assert excess_kurtosis(x) == pytest.approx(-2.0, abs=1e-12)


def test_excess_kurtosis_standard_error_matches_the_formula():
    assert excess_kurtosis_se(2500) == pytest.approx(np.sqrt(24 / 2500))
    assert excess_kurtosis_se(2500) == pytest.approx(0.098, abs=0.001)


def test_excess_kurtosis_is_unchanged_by_scaling_and_shifting():
    """Kurtosis is a shape statistic, so it must not depend on units."""
    x = rng(10).standard_normal(5000)
    assert excess_kurtosis(3 * x + 7) == pytest.approx(excess_kurtosis(x), abs=1e-10)


# --- calibration: the test with a known analytical answer -----------------
#
# Note 3, section 6. If H0 is true, the p-value is uniform on [0, 1] by
# construction, so the probability of rejecting at level alpha is exactly
# alpha. GBM satisfies H0 for all three tests, so each rejection rate must
# come out at alpha. The tolerance is derived, not guessed: the rejection
# count is Binomial(S, alpha), so the rate has standard error
# sqrt(alpha * (1 - alpha) / S), and 3 SE is the band.


N_CALIBRATION_SERIES = 1000
N_CALIBRATION_DAYS = 500
CALIBRATION_LEVEL = 0.05


def calibration_band(n_series, level, n_se=3):
    """The interval a correctly calibrated rejection rate should fall in."""
    se = np.sqrt(level * (1 - level) / n_series)
    return level - n_se * se, level + n_se * se


def gbm_series(n_series, n_days, seed):
    """Independent GBM series, one spawned generator each.

    SeedSequence.spawn gives independent streams rather than one stream cut
    into pieces, which is what "independent series" has to mean for the
    rejection rate to be a meaningful average.
    """
    children = np.random.SeedSequence(seed).spawn(n_series)
    return [
        gbm_log_returns(n_days, mu=0.0, sigma=0.01, rng=np.random.default_rng(child))
        for child in children
    ]


@pytest.fixture(scope="module")
def calibration_sample():
    return gbm_series(N_CALIBRATION_SERIES, N_CALIBRATION_DAYS, seed=20260902)


def test_calibration_band_is_derived_from_the_binomial_standard_error():
    low, high = calibration_band(1000, 0.05)
    assert low == pytest.approx(0.05 - 3 * np.sqrt(0.05 * 0.95 / 1000))
    assert (low, high) == pytest.approx((0.029, 0.071), abs=0.001)


def test_ljung_box_rejects_at_the_nominal_rate_on_a_true_null(calibration_sample):
    rate = np.mean(
        [ljung_box(r, lags=10)[1] < CALIBRATION_LEVEL for r in calibration_sample]
    )
    low, high = calibration_band(N_CALIBRATION_SERIES, CALIBRATION_LEVEL)
    assert low < rate < high, f"Ljung-Box rejection rate {rate} outside {low}-{high}"


def test_arch_lm_rejects_at_the_nominal_rate_on_a_true_null(calibration_sample):
    rate = np.mean(
        [arch_lm(r, lags=10)[1] < CALIBRATION_LEVEL for r in calibration_sample]
    )
    low, high = calibration_band(N_CALIBRATION_SERIES, CALIBRATION_LEVEL)
    assert low < rate < high, f"ARCH-LM rejection rate {rate} outside {low}-{high}"


def test_jarque_bera_rejects_at_the_nominal_rate_on_a_true_null(calibration_sample):
    """Jarque-Bera converges to its chi-squared limit slowly, so this uses a
    longer series than the other two. With n = 500 the test is known to be
    conservative, which would fail a band centred on alpha for the wrong
    reason -- a property of the asymptotics, not a bug in the code."""
    sample = gbm_series(N_CALIBRATION_SERIES, 5000, seed=20260903)
    rate = np.mean([jarque_bera(r)[1] < CALIBRATION_LEVEL for r in sample])
    low, high = calibration_band(N_CALIBRATION_SERIES, CALIBRATION_LEVEL)
    assert low < rate < high, f"Jarque-Bera rejection rate {rate} outside {low}-{high}"


def test_the_calibration_series_are_not_identical(calibration_sample):
    """A seeding bug that produced identical series would make every
    rejection rate 0 or 1, so check the streams really are distinct."""
    assert not np.array_equal(calibration_sample[0], calibration_sample[1])


# --- power: the tests must also detect what they are looking for ----------


def test_ljung_box_detects_autocorrelation():
    """A correctly calibrated test that never rejects is useless."""
    x = ar1(1000, phi=0.3, rng=rng(11))
    assert ljung_box(x, lags=10)[1] < 1e-6


def test_arch_lm_detects_volatility_clustering():
    r = garch11_log_returns(
        2000, mu=0.0, omega=2e-6, alpha=0.08, beta=0.90, rng=rng(12)
    )
    assert arch_lm(r, lags=10)[1] < 1e-3


def test_ljung_box_on_squared_returns_detects_clustering():
    """The same test, applied to r**2 instead of r. Note 3, section 1."""
    r = garch11_log_returns(
        2000, mu=0.0, omega=2e-6, alpha=0.08, beta=0.90, rng=rng(13)
    )
    assert ljung_box(r, lags=10)[1] > 0.01  # direction still unpredictable
    assert ljung_box(r**2, lags=10)[1] < 1e-3  # volatility is not


def test_jarque_bera_detects_heavy_tails():
    x = standardised_t(nu=5, size=2000, rng=rng(14))
    assert jarque_bera(x)[1] < 1e-6


# --- the summary used by the experiment script ---------------------------


def test_summarise_reports_every_test_at_every_lag():
    r = gbm_log_returns(1000, mu=0.0, sigma=0.01, rng=rng(15))
    out = summarise(r, lags=[10, 20], significance_level=0.05)

    for lag in (10, 20):
        for series in ("r", "r2", "abs_r"):
            assert f"lb_p_{series}_lag{lag}" in out
            assert f"lb_reject_{series}_lag{lag}" in out
        assert f"arch_p_lag{lag}" in out
    assert "jb_p" in out
    assert "excess_kurtosis" in out
    assert "acf1_r" in out


def test_summarise_returns_json_serialisable_values():
    """The summary goes into the run log, which refuses numpy scalars it
    cannot convert and must never silently drop a value."""
    import json

    r = gbm_log_returns(500, mu=0.0, sigma=0.01, rng=rng(16))
    out = summarise(r, lags=[10], significance_level=0.05)
    json.dumps(out)
    assert all(isinstance(v, (float, bool, int)) for v in out.values())


def test_summarise_reject_flags_agree_with_the_significance_level():
    r = garch11_log_returns(
        2000, mu=0.0, omega=2e-6, alpha=0.08, beta=0.90, rng=rng(17)
    )
    out = summarise(r, lags=[10], significance_level=0.05)
    assert out["lb_reject_r2_lag10"] == (out["lb_p_r2_lag10"] < 0.05)
    assert out["arch_reject_lag10"] == (out["arch_p_lag10"] < 0.05)


# --- input validation -----------------------------------------------------


def test_ljung_box_rejects_more_lags_than_observations():
    with pytest.raises(ValueError, match="lags"):
        ljung_box(rng().standard_normal(10), lags=20)


def test_autocorrelation_rejects_a_non_positive_lag():
    with pytest.raises(ValueError, match="max_lag"):
        autocorrelation(rng().standard_normal(100), max_lag=0)
