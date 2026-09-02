# Note 3 — Testing that a generated series has the properties we claim

> Working notes made during the build, with AI assistance. Study material,
> not part of the submitted report. Citations point to `docs/references.md`.

---

## 0. Why this note exists

Note 2 claims things about each generator: "Student-t gives heavy tails",
"GARCH gives clustering". Claiming is not evidence. If the report says the
control has property X, the repository must contain the measurement that
shows property X, produced by code with a seed and a log entry.

There is also a bug-catching purpose. A generator with a sign error, a
mis-scaled `σ`, or an off-by-one in the GARCH recursion will still produce a
plausible-looking price chart. It will not pass these tests. Eyeballing a
chart is not validation.

Three tests, one per claim. Each section says what the statistic is, what the
null hypothesis is, and how to read the output.

---

## 1. Ljung–Box test — is there autocorrelation?

**Used for two different claims**, which is the thing to keep straight:

- on returns `r_t` — we want **no** autocorrelation (fact a);
- on squared returns `r_t²` or absolute returns `|r_t|` — we want
  **strong** autocorrelation (fact c, volatility clustering).

Same test, opposite desired outcomes.

**The idea.** Compute the sample autocorrelation `ρ̂_k` at each lag
`k = 1 … m`. Any single one will be a bit off zero by chance. Ljung–Box
combines them into one statistic so we test all `m` lags at once instead of
peeking at each:

```
Q = n(n+2) · Σ_{k=1..m}  ρ̂_k² / (n − k)
```

Squaring means positive and negative correlations both count. Under the null
`Q` follows a chi-squared distribution with `m` degrees of freedom.

**H₀: `ρ_1 = ρ_2 = … = ρ_m = 0`** — the series is not autocorrelated up to
lag `m`.

**Reading it.** Large `Q`, small p-value → reject → autocorrelation is
present.

- On `r_t`: we *want* a large p-value. Failing to reject is what we are after.
- On `r_t²`: we *want* a tiny p-value. Rejecting hard is what we are after.

`statsmodels.stats.diagnostic.acorr_ljungbox(x, lags=[m], return_df=True)`.
Choosing `m`: too few lags misses slow-decaying clustering, too many dilutes
the statistic. Report a few values (say 10, 20, 50) rather than defending one
magic number, and put them in the config.

---

## 2. Engle's ARCH-LM test — is there volatility clustering?

Ljung–Box on `r²` already detects clustering. This test is the standard,
purpose-built one, and reporting both is cheap and more convincing.

**The idea.** If volatility clusters, today's squared return should be
predictable from recent squared returns. So regress `r_t²` on its own `m`
lags by ordinary least squares and ask whether the regression explains
anything. The statistic is

```
LM = n · R²
```

where `R²` comes from that auxiliary regression. Under the null it follows
chi-squared with `m` degrees of freedom. This is a **Lagrange multiplier**
test; the `n·R²` form is a general pattern for LM tests, which is why it
recurs across econometrics. Engle (1982).

**H₀: no ARCH effects** — `r_t²` is not linearly predictable from its own
lags, i.e. conditional variance is constant.

**Reading it.** Small p-value → reject → volatility clustering present.

- GBM and IID Student-t: expect a large p-value. **No** clustering.
- GARCH: expect a very small p-value. Clustering present.

`statsmodels.stats.diagnostic.het_arch(x, nlags=m)`. Returns
`(lm_stat, lm_pvalue, f_stat, f_pvalue)`; use the LM pair.

---

## 3. Jarque–Bera and excess kurtosis — are the tails heavy?

**The idea.** A normal distribution has skewness 0 and kurtosis 3. Measure
the sample skewness `S` and kurtosis `K` and see how far off they are:

```
JB = (n/6) · ( S² + (K − 3)²/4 )
```

Under normality, `JB` follows chi-squared with **2** degrees of freedom —
two, because two quantities are being tested at once. Jarque & Bera (1980).

**H₀: the data are normally distributed** (specifically, have the skewness
and kurtosis of a normal).

**Reading it.** Small p-value → reject normality.

- GBM: expect a **large** p-value. The returns really are Gaussian, so
  failing to reject is the correct result and confirms the generator works.
- Student-t and GARCH: expect a tiny p-value.

`scipy.stats.jarque_bera(x)`.

**Also report excess kurtosis directly, with its uncertainty.** A p-value
says "not normal"; it does not say *how* heavy the tails are, and the
magnitude is what matters for comparing the synthetic series to real ASX
data. Under normality the sample excess kurtosis has standard error roughly

```
SE ≈ √(24/n)
```

So for `n = 2500` daily observations, `SE ≈ 0.098`. An excess kurtosis of 4
is about 40 standard errors from zero — which also shows why the p-value is
uninformative here and the number itself is the useful output.

A caution for the Student-t generator: the `ν`th moment of a Student-t is
infinite for order ≥ ν, so kurtosis is only finite for `ν > 4`, and the
*sample* kurtosis of a t with small `ν` is unstable and grows with sample
size. With `ν` around 4 to 6 the numbers will be noisy across seeds. That is
a property of the distribution, not a bug, and it needs to be understood
before it is seen, or it will look like a broken generator.

---

## 4. Expected outcomes — the acceptance criteria

This table is the specification for the validation module. `p_LB(r)` is the
Ljung–Box p-value on returns, and so on.

| Generator      | `p_LB(r)` | `p_LB(r²)` | `p_ARCH` | `p_JB`  | excess kurtosis |
| -------------- | :-------: | :--------: | :------: | :-----: | :-------------: |
| GBM            |  large    |   large    |  large   | large   | ≈ 0             |
| Student-t IID  |  large    |   large    |  large   | tiny    | large, positive |
| GARCH(1,1)     |  large    |   tiny     |  tiny    | tiny    | positive        |
| Real ASX data  |  large    |   tiny     |  tiny    | tiny    | positive        |

"Large" means "do not reject at the chosen α"; "tiny" means "reject
decisively". The bottom row is the target the ladder is climbing towards, and
running the identical validation on real data is what makes the comparison
meaningful rather than decorative.

Note the top-left cell. For **every** generator, `p_LB(r)` must be large.
That column is the null-model requirement from Note 1. If it fails, the
generator is not a valid control and nothing downstream is trustworthy.

---

## 5. Reading p-values honestly

Four things to have straight, because each is a likely interview question.

**A large p-value is not proof.** Failing to reject H₀ means the data did not
provide enough evidence against it, not that H₀ is true. The correct phrasing
throughout the report is "no significant autocorrelation was detected", never
"the series was proven to have no autocorrelation". For GBM we happen to know
H₀ is true because we constructed it that way — but then the test is checking
the *code*, not the mathematics.

**Large `n` makes trivial effects significant.** With `n = 2500`, an
autocorrelation of 0.04 — economically meaningless, nothing tradeable —
can reject at the 5% level. So report the **effect size** alongside the
p-value: the actual `ρ̂_k` values, not just "p < 0.05". This point cuts both
ways and it is exactly the kind of nuance that separates a strong project
from a mechanical one.

**Multiple testing is here too.** Running three tests on three generators
across many seeds is dozens of tests. At α = 0.05, roughly one in twenty
rejects by chance even when everything is correct. Since the entire project is
*about* multiple testing, it would be embarrassing to fall into it in the
validation. Two defences: report the distribution of p-values over seeds
rather than a single run's verdict, and treat the calibration check below as
the real evidence.

**Validate over many seeds, not one series.** A single series can pass or
fail by luck. Generate `S` independent series, run the tests on each, and
report the *rejection rate*. That is a stable quantity and it leads directly
to a test with a known answer.

---

## 6. The calibration check — a test with an analytical answer

`CLAUDE.md` requires that any function doing statistical work has a test
against a known analytical answer. Here is the one for this module, and it is
a genuinely good check rather than a box-tick.

**The known answer.** If H₀ is true and the test is correctly calibrated,
then by the definition of a p-value the p-value is uniformly distributed on
`[0, 1]`. So the probability of rejecting at level α is exactly α.

**The test.** Generate `S = 1000` independent GBM series. Run Ljung–Box on
the returns of each. The fraction with `p < 0.05` should be about 0.05.

**The tolerance, derived rather than guessed.** The count of rejections is
Binomial(`S`, 0.05), so the rejection rate has standard error
`√(0.05 · 0.95 / 1000) ≈ 0.0069`. A ±3 SE band is roughly `[0.029, 0.071]`.
Assert the rate falls inside that. The bound is computed from `S` and `α` in
the test itself, not hardcoded — so it stays correct if `S` changes, and the
derivation is defensible under questioning.

This single test catches a remarkable amount: a broken GBM generator, a
misused statsmodels call, a wrong lag convention, a seeding bug that makes
the "independent" series identical. If the rejection rate comes out at 0.35
or 0.001, something is wrong, and the test says so without anyone needing to
inspect a chart.

The same check applies to Jarque–Bera on GBM returns, which is a second
independent probe of the same generator.

---

## Explain these back before we write the validation module

1. For each of the three tests, state the null hypothesis in one sentence
   and say whether we want a large or small p-value on a GBM series.
2. Ljung–Box appears twice with opposite desired outcomes. Explain what is
   being tested each time and why both matter.
3. Why is reporting only "p < 0.05" insufficient when `n = 2500`? What
   should be reported alongside it?
4. Explain the calibration check: what is the known analytical answer, where
   does the ±3 SE tolerance come from, and name two bugs it would catch.
