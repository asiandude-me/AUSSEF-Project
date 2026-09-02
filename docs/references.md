# References

Every method taken from the literature is listed here, with a note on where
it is used in this project. Acknowledged derivation is expected; concealed
derivation is not. Add to this file whenever a paper informs a design
decision, at the time the decision is made.

Format: full citation, then a line on the role it plays here.

---

## Data snooping and multiple testing

**White, H. (2000). A Reality Check for Data Snooping.** *Econometrica*,
68(5), 1097–1126. https://doi.org/10.1111/1468-0262.00152
Introduces the Reality Check: a bootstrap procedure that tests whether the
best model out of many outperforms a benchmark, correcting for the fact that
the best of N was selected by searching. One of the two corrections applied in
step 7 of the design.

**Sullivan, R., Timmermann, A., & White, H. (1999). Data-Snooping,
Technical Trading Rule Performance, and the Bootstrap.** *Journal of
Finance*, 54(5), 1647–1691. https://doi.org/10.1111/0022-1082.00163
Applies White's Reality Check to a universe of technical trading rules on the
Dow. The closest published antecedent to this project's question, and the
paper to cite when explaining what is already known versus what this project
adds. Their universe of rules is a useful template for the parameter grid.

**Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.**
*Journal of Portfolio Management*, 40(5), 94–107.
https://doi.org/10.3905/jpm.2014.40.5.094
Gives the Deflated Sharpe Ratio, which adjusts an observed Sharpe for the
number of trials, the variance of the trial Sharpes, the sample length, and
the skewness and kurtosis of returns. The second correction applied in step 7.
Its dependence on skewness and kurtosis is the reason the choice of synthetic
null model matters (Note 2).

**Harvey, C. R., & Liu, Y. (2015). Backtesting.** *Journal of Portfolio
Management*, 42(1), 13–28. https://doi.org/10.3905/jpm.2015.42.1.013
Practical treatment of haircutting reported Sharpe ratios for multiple
testing. Useful for framing the size of the correction and for the discussion.

**Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the Cross-Section of
Expected Returns.** *Review of Financial Studies*, 29(1), 5–68.
https://doi.org/10.1093/rfs/hhv059
Argues that the conventional significance threshold is far too lenient once
the number of tested factors is counted. Background and motivation.

**Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014).
Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest
Overfitting on Out-of-Sample Performance.** *Notices of the AMS*, 61(5),
458–471. https://doi.org/10.1090/noti1105
Accessible account of how in-sample backtest performance degrades out of
sample. Directly relevant to step 6 of the design.

---

## Market efficiency and the null model

**Samuelson, P. A. (1965). Proof That Properly Anticipated Prices Fluctuate
Randomly.** *Industrial Management Review*, 6(2), 41–49.
The martingale argument underlying the null model: if prices already reflect
available information, price changes are unforecastable from that information.
Cited in Note 1 for the definition of "no exploitable structure".

**Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and
Empirical Work.** *Journal of Finance*, 25(2), 383–417.
https://doi.org/10.2307/2325486
The efficient markets framework and the weak-form hypothesis, which is the
hypothesis technical trading rules test.

**Brock, W., Lakonishok, J., & LeBaron, B. (1992). Simple Technical Trading
Rules and the Stochastic Properties of Stock Returns.** *Journal of Finance*,
47(5), 1731–1764. https://doi.org/10.1111/j.1540-6261.1992.tb04681.x
Reports apparent profitability of moving-average and trading-range-break rules
on the Dow. Source of the moving-average crossover rules used here, and the
paper Sullivan, Timmermann & White later re-examined for data snooping. The
pair together is the story this project reproduces in miniature.

---

## Properties of financial returns

**Cont, R. (2001). Empirical Properties of Asset Returns: Stylized Facts and
Statistical Issues.** *Quantitative Finance*, 1(2), 223–236.
https://doi.org/10.1080/713665670
The catalogue of stylised facts used in Note 2 and the source of the
acceptance criteria for the synthetic generators.

**Mandelbrot, B. (1963). The Variation of Certain Speculative Prices.**
*Journal of Business*, 36(4), 394–419. https://doi.org/10.1086/294632
Early documentation of heavy tails and volatility clustering in price series.

**Lo, A. W. (2002). The Statistics of Sharpe Ratios.** *Financial Analysts
Journal*, 58(4), 36–52. https://doi.org/10.2469/faj.v58.n4.2453
Derives the sampling distribution of the Sharpe ratio and shows how the
standard errors change when returns are not IID. The basis for the argument in
Note 2 that a GBM null gives a null distribution that is too narrow.

---

## Models used for the synthetic control

**Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with
Estimates of the Variance of United Kingdom Inflation.** *Econometrica*,
50(4), 987–1007. https://doi.org/10.2307/1912773
Introduces ARCH and the Lagrange multiplier test for ARCH effects used in
Note 3.

**Bollerslev, T. (1986). Generalized Autoregressive Conditional
Heteroskedasticity.** *Journal of Econometrics*, 31(3), 307–327.
https://doi.org/10.1016/0304-4076(86)90063-1
Introduces GARCH. The GARCH(1,1) specification is the third synthetic
generator.

**Sheppard, K., et al. `arch`: Autoregressive Conditional Heteroskedasticity
(ARCH) and other tools for financial econometrics (Python package).**
https://bashtage.github.io/arch/
The implementation used for GARCH simulation and fitting.

---

## Statistical tests

**Ljung, G. M., & Box, G. E. P. (1978). On a Measure of Lack of Fit in Time
Series Models.** *Biometrika*, 65(2), 297–303.
https://doi.org/10.1093/biomet/65.2.297
The portmanteau autocorrelation test used in Note 3, applied to returns and to
squared returns.

**Jarque, C. M., & Bera, A. K. (1980). Efficient Tests for Normality,
Homoscedasticity and Serial Independence of Regression Residuals.**
*Economics Letters*, 6(3), 255–259.
https://doi.org/10.1016/0165-1765(80)90024-5
The normality test used in Note 3.

---

## Software

Cited because results depend on them and versions are pinned in
`requirements-lock.txt`.

- **NumPy** — Harris, C. R., et al. (2020). Array programming with NumPy.
  *Nature*, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2
- **SciPy** — Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms
  for scientific computing in Python. *Nature Methods*, 17, 261–272.
  https://doi.org/10.1038/s41592-019-0686-2
- **pandas** — McKinney, W. (2010). Data structures for statistical computing
  in Python. *Proc. 9th Python in Science Conf.*, 56–61.
  https://doi.org/10.25080/Majora-92bf1922-00a
- **statsmodels** — Seabold, S., & Perktold, J. (2010). statsmodels:
  Econometric and statistical modeling with Python. *Proc. 9th Python in
  Science Conf.* https://www.statsmodels.org/
- **Matplotlib** — Hunter, J. D. (2007). Matplotlib: A 2D graphics
  environment. *Computing in Science & Engineering*, 9(3), 90–95.
  https://doi.org/10.1109/MCSE.2007.55

---

## Data source

**yfinance** (Python package). https://github.com/ranaroussi/yfinance
Used to download ASX price history. **Outstanding task:** read and record the
terms of use for the underlying data before the first download, and note here
what they permit for a non-commercial student research project. Do not fetch
data until this is done.
