# Note 1 — Why geometric Brownian motion is the right null model

> Working notes made during the build, with AI assistance, to get the ideas
> straight before writing any generator code. They are study material, not
> part of the submitted report. Citations point to `docs/references.md`.

---

## 1. Returns, not prices

Let `P_t` be the closing price on day `t`. The **log-return** is

```
r_t = ln(P_t / P_{t-1}) = ln P_t - ln P_{t-1}
```

Two reasons the whole project works in log-returns rather than prices.

**They add.** The log-return over two days is `ln(P_t/P_{t-2})`, which is
exactly `r_t + r_{t-1}`. Simple returns `(P_t - P_{t-1})/P_{t-1}` do not add;
they compound, so a two-day simple return is `(1+g_t)(1+g_{t-1}) - 1`.
Adding is what lets a sum of daily returns be a strategy's total return, and
what makes the central limit theorem usable on returns.

**They are scale-free.** A 1% move is the same size in log-returns whether
the price is $2 or $200. Prices are not: a $1 move means something different
at each level.

Prices come back by exponentiating the cumulative sum:

```
P_t = P_0 · exp(r_1 + r_2 + ... + r_t)
```

Because `exp` of anything is positive, a price built this way can never go
negative. Hold onto that; it is the argument in section 4.

---

## 2. What geometric Brownian motion is, in one line

In discrete time, GBM is exactly this statement about log-returns:

```
r_t = μ + σ ε_t ,     ε_t ~ N(0, 1), independent across t
```

That is it. Independent, identically distributed normal log-returns.
"Geometric" refers to the exponentiation in section 1: the *log* of price
takes a random walk with drift, so the price itself moves multiplicatively.

Two parameters. `μ` is the mean daily log-return (the drift), `σ` is the
daily volatility. Both are constants — they do not depend on `t` and they do
not depend on anything that has already happened.

Note the drift correction. If you want the price to grow at expected simple
rate `m` per day, you must set `μ = m − σ²/2`, not `μ = m`. The gap comes
from Jensen's inequality: `E[exp(X)] > exp(E[X])` for a non-degenerate `X`.
Averaging *after* exponentiating is not the same as exponentiating the
average. This is a real trap when calibrating the generator to a target
return, and it is exactly the kind of thing an interviewer asks about.

---

## 3. What "no exploitable structure" means, precisely

This is the definition the whole experiment rests on, so it is worth stating
carefully rather than waving at "randomness".

Write `F_{t}` for everything observable up to and including day `t`: the
whole history of prices and returns. The condition is

```
Distribution of r_{t+1} given F_t  =  Distribution of r_{t+1}
```

In words: **knowing the entire past tells you nothing you did not already
know about tomorrow.** The conditional distribution equals the unconditional
one. Under GBM this holds by construction, because the `ε_t` are independent.

Now watch why that kills every technical trading rule at once.

A technical rule is a function that reads the past and outputs a position.
Let `s_t` be the position held into day `t+1`: `+1` for long, `0` for flat,
`−1` for short, or anything in between. The defining feature of a
*technical* rule is that `s_t` is a function of `F_t` only — moving averages,
breakouts, momentum, all of them look backwards. The rule's return on day
`t+1` is `s_t · r_{t+1}`.

Its expected value, using the tower property and then independence:

```
E[s_t · r_{t+1}]  =  E[ E[ s_t · r_{t+1} | F_t ] ]      (tower property)
                  =  E[ s_t · E[ r_{t+1} | F_t ] ]      (s_t is known given F_t)
                  =  E[ s_t · μ ]                        (independence: the
                                                          conditional mean is
                                                          just μ)
                  =  μ · E[s_t]
```

The result depends on the rule **only through `E[s_t]`**, the average
exposure. Not through the timing. Not through the cleverness. A rule that is
long 60% of the time earns `0.6μ` in expectation, whether it picks those days
by a moving-average crossover or by flipping a coin. No amount of searching
changes that, because it is a statement about the data-generating process,
not about the search.

So on a GBM series, every difference in performance between rules is sampling
noise. Nothing else is available. When we run 10,000 rules and keep the best
Sharpe ratio, we are computing **the maximum of 10,000 noisy estimates of the
same underlying zero**. That maximum is not close to zero, and *how far from
zero it is* is the quantity this project measures. It is the entire point.

There is a name for this in the literature: the price is a **martingale**
(after adjusting for drift). Samuelson (1965) and Fama (1970) are the
canonical references; Sullivan, Timmermann & White (1999) apply the
data-snooping argument to technical trading rules specifically.

### The drift caveat — flag this now

Look again at `μ · E[s_t]`. If `μ > 0` and the rule is long-only, then
`E[s_t] > 0` and the rule makes money in expectation. It has no *skill*; it is
harvesting drift while being out of the market part of the time.

This matters for the design. Two consequences to build in later:

1. Every strategy must be scored against **buy-and-hold on the same series**,
   not against zero. Buy-and-hold has `s_t ≡ 1`, the maximum possible drift
   harvest.
2. Consider setting `μ = 0` in the baseline synthetic series, or reporting
   both. With zero drift the null is unambiguous: any positive Sharpe is
   purely search artefact, no interpretation needed.

Decide this before running anything, and record the decision in `CLAUDE.md`.

---

## 4. What breaks with a simpler model

Suppose we skipped log-returns and put IID Gaussian noise directly on price:

```
P_t = P_{t-1} + ε_t ,   ε_t ~ N(0, σ²)
```

an arithmetic random walk. Two things break, and the second is fatal for this
project.

**Prices can go negative.** `P_t` is a sum of normals, so it is normal, so it
has positive probability of being below zero. A negative price is not a
price. Worse, log-returns are then undefined for part of the series, so the
strategy code crashes or silently drops data — and any fix is a place where a
judge asks "what did you do with those, and why?"

**Volatility becomes predictable from the price level — which is structure.**
Under the arithmetic walk, `σ` is a fixed number of *dollars*. So the
percentage move, `ε_t / P_{t-1}`, shrinks as the price rises and grows as it
falls. A rule that says "trade bigger when the price is low" would then be
detecting something genuinely present in the data.

That is a catastrophe for a *control* group. The control's job is to contain
provably zero exploitable structure, so that any signal the search reports is
known to be an artefact of searching. If the control contains real structure,
the baseline is contaminated and the comparison in step 5 of the design means
nothing. Under GBM the percentage move is `σ` regardless of the price level,
so this leak does not exist.

Two smaller points worth having ready:

- **Shuffling real returns** is a tempting alternative null: permute the
  actual returns, keeping the exact distribution and destroying the order.
  It is a legitimate method (a bootstrap) and gives fat tails for free. But
  permuting destroys volatility clustering too, so it is not obviously more
  realistic than GBM, and it makes the null depend on the real series being
  tested. Worth mentioning as an alternative considered; possibly worth
  adding later as a robustness check.
- **The normal assumption is doing less work than it looks.** The martingale
  argument in section 3 needs only that `E[r_{t+1} | F_t] = μ`. It never uses
  normality. That is why swapping in Student-t innovations, in the next note,
  changes the tails without breaking the null. The series stays unpredictable.

---

## Explain these back before we write the generator

1. Write down the expected return of an arbitrary technical rule on a GBM
   series and derive `μ · E[s_t]` from scratch, saying which step uses
   independence and which uses that `s_t` is known from the past.
2. A judge says: "Your synthetic series has a positive drift and your best
   strategy made money. Doesn't that mean it found something?" Answer them.
3. Explain why an arithmetic random walk on price would corrupt the control
   group, using the words "percentage volatility" and "price level".
4. Why does `μ = m − σ²/2` rather than `μ = m`, and where does the `σ²/2`
   come from?
