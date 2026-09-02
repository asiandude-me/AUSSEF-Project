# Synthetic generator validation

500 independent series per generator, 2500 days each, tested at the 0.05 level with Ljung-Box and ARCH-LM at lag 10.

Each cell is the **fraction of series rejecting** that test. Where the
acceptance table in `docs/notes/03_validation_tests.md` says the p-value
should be large, a correct generator gives a rate near 0.05; where it
says tiny, a rate near 1.

A correctly calibrated rate has standard error 0.0097, so anything in
roughly 0.021 to 0.079 is consistent with 0.05.

| Generator | reject LB(r) | **reject LB(r) robust** | reject LB(r²) | reject LB(&#124;r&#124;) | reject ARCH | reject JB | median excess kurtosis (IQR) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gbm | 0.042 | **0.044** | 0.060 | 0.050 | 0.056 | 0.046 | -0.01 (0.12) |
| student_t | 0.032 | **0.034** | 0.070 | 0.026 | 0.070 | 1.000 | 3.37 (2.23) |
| garch11 | 0.256 | **0.056** | 1.000 | 1.000 | 1.000 | 0.982 | 0.80 (0.66) |
| garch11_t | 0.494 | **0.046** | 1.000 | 1.000 | 1.000 | 1.000 | 5.90 (4.53) |

## How to read the first two columns

`reject LB(r)` is the null-model requirement. Every generator must sit
near 0.05 here: the returns themselves carry no predictable direction,
which is what makes the series a valid control for a directional trading
rule. A generator failing this column is broken as a control, and that is
a correctness bug rather than a cosmetic one.

**The two versions of that column disagree on the GARCH rows, and the
robust one is the correct one.** The classical Ljung-Box test assumes the
data are independent under the null and uses 1/n as the variance of each
sample autocorrelation. GARCH returns are serially uncorrelated but not
independent, so their sample autocorrelations are more variable than that,
and a test built on the wrong null variance rejects far too often. The
returns are unpredictable in direction by construction: `E[r_t | past] = mu`
holds exactly for every generator here. The robust column corrects the
variance estimate (Diebold 1986) and recovers the nominal rate.

This is worth reporting rather than quietly fixing. It is a small worked
example of the project's own thesis: a standard test, applied outside the
assumptions it was derived under, reports structure that is not there.

The remaining columns are what separates the generators, and they are
meant to differ: heavy tails show up in `reject JB`, volatility clustering
in `reject LB(r²)` and `reject ARCH`.

## Notes on the measured effect sizes

Two numbers in the kurtosis column are smaller or noisier than their
closed forms, for reasons that are properties of the processes rather
than defects in the generators. Both are worth having ready.

**GARCH.** The closed form for `alpha = 0.08, beta = 0.90` gives an excess
kurtosis of 1.43, but the median measured over 2500-day series is lower.
The fourth moment exists only when `3a^2 + 2ab + b^2 < 1`, and these
parameters satisfy that by 0.027. Convergence is correspondingly slow: the
median sample estimate climbs from about 0.80 at n = 2500 to 1.41 at
n = 2,000,000. Ten years of daily data is simply too short to see the
asymptotic value. `alpha + beta = 0.98` is what makes the clustering
realistically persistent, so this is a consequence of using realistic
parameters, not a reason to change them.

**Student-t.** At `nu = 5` the theoretical excess kurtosis is `6/(nu-4) = 6`,
and the measured median is lower with a wide inter-quartile range. The
sample kurtosis of a Student-t is unstable for small `nu` and grows with
sample size, as flagged in `docs/notes/03_validation_tests.md`.

In both cases the test-based columns, which do not depend on estimating a
fourth moment, are unaffected.

## Rejection rates at every lag

| generator | n_series | reject_lb_r_lag10 | reject_lb_r2_lag10 | reject_lb_abs_r_lag10 | reject_lb_robust_r_lag10 | reject_arch_lag10 | reject_lb_r_lag20 | reject_lb_r2_lag20 | reject_lb_abs_r_lag20 | reject_lb_robust_r_lag20 | reject_arch_lag20 | reject_lb_r_lag50 | reject_lb_r2_lag50 | reject_lb_abs_r_lag50 | reject_lb_robust_r_lag50 | reject_arch_lag50 | reject_jb | median_excess_kurtosis | iqr_excess_kurtosis | mean_acf1_r | mean_acf1_r2 | median_std | median_p_lb_r_lag10 | median_p_lb_r2_lag10 | median_p_arch_lag10 | median_p_jb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gbm | 500 | 0.0420 | 0.0600 | 0.0500 | 0.0440 | 0.0560 | 0.0520 | 0.0460 | 0.0420 | 0.0480 | 0.0520 | 0.0680 | 0.0600 | 0.0480 | 0.0640 | 0.0560 | 0.0460 | -0.0107 | 0.1212 | -0.0001 | -0.0012 | 0.0100 | 0.4750 | 0.4961 | 0.4852 | 0.5084 |
| student_t | 500 | 0.0320 | 0.0700 | 0.0260 | 0.0340 | 0.0700 | 0.0420 | 0.0860 | 0.0380 | 0.0400 | 0.0840 | 0.0480 | 0.0900 | 0.0560 | 0.0480 | 0.0900 | 1.0000 | 3.3708 | 2.2259 | -0.0006 | -0.0000 | 0.0100 | 0.5179 | 0.7341 | 0.7282 | 0.0000 |
| garch11 | 500 | 0.2560 | 1.0000 | 1.0000 | 0.0560 | 1.0000 | 0.3080 | 1.0000 | 1.0000 | 0.0600 | 1.0000 | 0.3740 | 1.0000 | 1.0000 | 0.0640 | 1.0000 | 0.9820 | 0.7956 | 0.6626 | 0.0002 | 0.1700 | 0.0099 | 0.1847 | 0.0000 | 0.0000 | 0.0000 |
| garch11_t | 500 | 0.4940 | 1.0000 | 1.0000 | 0.0460 | 1.0000 | 0.6060 | 1.0000 | 1.0000 | 0.0460 | 1.0000 | 0.6360 | 1.0000 | 1.0000 | 0.0600 | 0.9960 | 1.0000 | 5.9046 | 4.5333 | -0.0015 | 0.1278 | 0.0095 | 0.0506 | 0.0000 | 0.0000 | 0.0000 |
