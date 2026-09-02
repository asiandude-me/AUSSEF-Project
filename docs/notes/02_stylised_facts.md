# Note 2 — Stylised facts, where GBM fails, and why that threatens the conclusion

> Working notes made during the build, with AI assistance. Study material,
> not part of the submitted report. Citations point to `docs/references.md`.

---

## 1. What a "stylised fact" is

Real return series from different assets, countries and decades share a set
of statistical properties that show up almost everywhere. They are called
**stylised facts** because they are robust empirical regularities rather than
consequences of any particular theory. Cont (2001) is the standard catalogue
and is the reference to cite.

The four that matter here.

**(a) Returns have essentially no linear autocorrelation.** `corr(r_t,
r_{t+k})` is near zero for every lag `k ≥ 1`, beyond very short intraday
horizons. This is the fact that makes markets look unpredictable, and it is
the one directly connected to the null model in Note 1.

**(b) Heavy tails.** The distribution of daily returns has far more
probability in the extremes than a normal distribution does. Measured by
**excess kurtosis** (kurtosis minus 3, so that a normal scores 0), daily
equity returns typically land somewhere around 3 to 10. Large moves are not
rare curiosities; they are a regular feature.

**(c) Volatility clustering.** Big moves are followed by big moves, of either
sign; quiet periods are followed by quiet periods. Formally: `r_t` itself is
uncorrelated, but `|r_t|` and `r_t²` are strongly and *persistently*
autocorrelated, with correlations still visible at lags of weeks or months.
This is the single most important departure from GBM. Mandelbrot noticed it
in 1963; Engle (1982) and Bollerslev (1986) built the models.

**(d) Aggregational Gaussianity.** As you lengthen the return period from
daily to weekly to monthly, the distribution looks progressively more normal.
Relevant mostly as a reminder that "returns are not normal" is a statement
about a time scale.

There is also the **leverage effect**: negative returns raise future
volatility more than positive returns of the same size do. Worth naming so it
is clear it was considered, but modelling it needs an asymmetric GARCH
variant (GJR or EGARCH) and that is extra machinery this project does not
need. If asked, the honest answer is: it is a known asymmetry, we did not
model it, and here is why it does not change the conclusion.

---

## 2. What each candidate generator reproduces

|                              | (a) no autocorr. in `r` | (b) heavy tails | (c) vol. clustering |
| ---------------------------- | :---------------------: | :-------------: | :-----------------: |
| GBM (Gaussian IID)           |           yes           |       no        |         no          |
| IID Student-t innovations    |           yes           |       yes       |         no          |
| GARCH(1,1), Gaussian innov.  |           yes           |    yes\*        |         yes         |
| GARCH(1,1), t innovations    |           yes           |       yes       |         yes         |

\* This is the subtle and interesting one. GARCH with *perfectly normal*
innovations still produces heavy tails in the returns you observe. The
mechanism: the observed return is `r_t = μ + σ_t ε_t`, where `σ_t` itself
varies over time. Mixing normals with different variances produces a
distribution with more mass in the tails than any single normal — the calm
days pile up near the centre while the volatile days reach far out. So
clustering *generates* excess kurtosis for free. Two apparently separate
stylised facts turn out to have one cause.

Every row in that table satisfies (a). That is not a coincidence and it is
the key point of the whole note, so it gets its own section.

---

## 3. The point that must survive the interview

**All four generators are unpredictable in direction. None of them is
predictable in the sense a trading rule needs.**

Note 1 showed that a technical rule's expected return is `μ · E[s_t]`, and
the only property used was `E[r_{t+1} | F_t] = μ`. Check that condition for
GARCH(1,1):

```
r_t = μ + σ_t ε_t ,    σ_t² = ω + α (r_{t-1} − μ)² + β σ_{t-1}²
```

`σ_t` is computed entirely from the past, so given `F_{t-1}` it is a known
number. `ε_t` is independent with mean zero. Therefore

```
E[r_t | F_{t-1}] = μ + σ_t · E[ε_t] = μ
```

Exactly as under GBM. The conditional *variance* moves around, and it is
genuinely forecastable — that is what GARCH is for. The conditional *mean*
does not move at all.

So: **volatility is predictable, direction is not.** A moving-average
crossover rule bets on direction. It has no edge on any of these series. The
null holds across the entire ladder of generators, and the ladder varies only
the realism of the noise, not the presence of signal. That is precisely what a
control group should do.

A sharp interviewer might push: could a rule exploit predictable volatility?
It could size positions better, changing the *risk* profile — but the Sharpe
ratio of a directional rule still has zero expected numerator, because the
expected excess return is zero however you scale it. Worth being ready to say
this cleanly.

---

## 4. Why GBM's failures matter *for this experiment specifically*

The measured quantity is:

> the distribution of "best in-sample Sharpe found by searching N rules",
> when the data contains no signal.

Call that the **null distribution**. Everything in step 5 of the design
compares the real-data result against it. So the question is not "is GBM
realistic?" in the abstract. It is: **does using GBM instead of something more
realistic change the shape of that null distribution?**

It does, and in a specific and dangerous direction.

The Sharpe ratio is `mean(returns) / sd(returns)`, a *statistic estimated
from a finite sample*. Like any estimate it has a sampling distribution, and
the width of that distribution controls how large the best-of-N can get. Two
mechanisms make it wider under realistic returns than under GBM:

**Heavy tails.** With fat-tailed returns, a handful of extreme days dominate
both the mean and the standard deviation. A rule that happens to be positioned
correctly on three big days posts a high Sharpe on the strength of those three
days. Under Gaussian returns those days do not exist, so that particular route
to a lucky high Sharpe is closed off.

**Volatility clustering.** Sharpe's standard error is derived assuming
independent, identically distributed returns. Real returns violate this, and
Lo (2002) works through what that does to the sampling distribution: the
estimator is noisier than the IID formula says. Bailey & López de Prado (2014)
build the Deflated Sharpe Ratio around exactly this, using skewness and
kurtosis as inputs — which is a strong hint that the correction we plan to
apply in step 7 *cares* about the properties GBM lacks.

Both push the same way: **the null distribution is too narrow under GBM.**

### The concrete risk, spelled out

The p-value in step 5 is roughly "what fraction of synthetic series produced a
best Sharpe at least as large as the real one". If the synthetic null is too
narrow, that fraction comes out too small, and:

- the real-data result looks significant when it is not;
- the project reports a **false positive** — it concludes real markets contain
  detectable technical signal, when what actually happened is that the control
  was too easy;
- the conclusion is exactly backwards from the project's own thesis, which is
  the most embarrassing possible way for this to fail.

This is worth being blunt about in the report. A weak control biases the
finding *towards* apparent signal. Since the project's whole argument is that
apparent signal is an artefact, a contaminated control would let a critic say
the headline result was manufactured by the null being generous.

### Which turns the weakness into the method

Because the failure direction is known, the ladder of generators is not three
attempts at one thing. It is a **sensitivity analysis**:

- GBM — the most conservative (narrowest) null. Establishes the mechanism in
  the simplest possible setting.
- Student-t — adds heavy tails only. Isolates the tail contribution.
- GARCH(1,1) — adds clustering, and heavy tails come along with it.

Then report the p-value under each. Two possible outcomes and both are
publishable:

- The conclusion is stable across all three. Then it is robust to the realism
  of the null, and saying so is much stronger than any single number.
- The conclusion flips — significant under GBM, not significant under GARCH.
  Then the finding is that *the choice of null model determines the answer*,
  which is itself a sharp result about data-snooping methodology and directly
  relevant to anyone using these tests.

Either way, running all three and reporting the comparison is the defensible
move. Running only GBM and hoping is not.

---

## 5. What this implies for the generator module

Falls out directly, and these become the acceptance criteria in Note 3:

1. Three variants behind one interface, so the identical search pipeline runs
   over each without modification.
2. Every variant must be **validated before use** — the properties are claimed
   for it, so they must be measured, not assumed. Note 3 covers the tests.
3. Every variant must satisfy fact (a). If a generator produces
   autocorrelated returns, it is broken as a control, and that is a
   correctness bug, not a cosmetic one.
4. GARCH parameters need choosing. Typical fitted daily equity values are
   around `α ≈ 0.05–0.1`, `β ≈ 0.85–0.9`. The constraint `α + β < 1` is
   required for the variance to be stationary; as `α + β → 1` clustering
   becomes more persistent. `ω` then sets the long-run variance,
   `ω / (1 − α − β)`. These are config values, never literals in the code.
5. Fitting GARCH to real ASX data to pick the parameters would make the
   control calibrated to the treatment. That is defensible and arguably
   better, but it is a design decision with consequences — decide it
   deliberately and write it in `CLAUDE.md`.

---

## Explain these back before we write the generator

1. Name the three stylised facts that matter here, and say which of them GBM
   reproduces.
2. Explain why a GARCH process with Gaussian innovations still produces
   excess kurtosis in the observed returns.
3. A judge says: "Your control is a Gaussian random walk. Real markets have
   fat tails. Doesn't that make your comparison invalid?" Answer, and say
   which direction the bias runs and what you did about it.
4. Why is a GARCH series still a valid null for this experiment, given that
   something about it is genuinely forecastable? Use the phrase "conditional
   mean".
